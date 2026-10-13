"""LoRA fine-tuning for an Arabic line-level OCR VLM (Task 6).

Trains a LoRA adapter on Qwen2.5-VL from image+text line pairs produced for our
corpus, then saves MERGED weights so `hf_vlm_engine.py` can load them with a plain
`from_pretrained` (PEFT adapters are not auto-applied by the engine).

Data layout (default --data annotation/):
    annotation/images/<id>.png     # a single text line
    annotation/text/<id>.txt       # the exact transcription of that line

Usage:
    # 1. Validate the dataset without loading a model:
    .venv/bin/python train_lora.py --check-data --data annotation/

    # 2. Train (requires: pip install -U peft; torch+transformers already present):
    .venv/bin/python train_lora.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --data annotation/ \
        --out models/qwen-arabic-lora \
        --epochs 3 --accum 8

    # 3. Evaluate through the harness:
    VISION_HF_MODEL=./models/qwen-arabic-lora \
        .venv/bin/python eval_ocr.py --engine hf \
        --cache-dir probe/_ocr_cache/lora --report probe/eval_report_lora.json --force

Notes:
  - Batch size is fixed at 1 per forward (line images vary in size); use --accum
    for an effective batch size. This keeps the collator simple and robust.
  - Saves a PEFT adapter alongside the merged model for reproducibility.
  - GPU: set CUDA_VISIBLE_DEVICES if you have several. Training saturates the GPU;
    do not run pipeline OCR at the same time (thermal guard does not cover training).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from ocr_engine import _LINE_PROMPT
from text_cleanup import _is_diacritic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_lora")

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Training prompts. _LINE_PROMPT is shared with the engines' `ocr_line` so the
# fine-tuned model sees the instruction it was trained on. _PLAIN_PROMPT goes with
# --strip-diacritics, whose targets carry no harakat.
_PLAIN_PROMPT = (
    "Transcribe the Arabic text in this line image. Output the letters only, "
    "without diacritical marks (harakat). Output only the transcribed text."
)
_TATWEEL = "\u0640"


def strip_marks(text: str) -> str:
    """Drop harakat/Quranic marks and tatweel; letters, digits, punctuation stay."""
    return "".join(ch for ch in text if not _is_diacritic(ch) and ch != _TATWEEL)


def load_target(txt: Path, strip: bool) -> str:
    text = txt.read_text(encoding="utf-8").strip()
    return strip_marks(text) if strip else text


# --- Data discovery ---------------------------------------------------------


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, transcript_path) pairs matched by file stem.

    Looks for `images/` and `text/` subdirectories first; otherwise scans the
    given directory for images with a sibling `<stem>.txt`.
    """
    pairs: list[tuple[Path, Path]] = []
    img_dir = data_dir / "images"
    txt_dir = data_dir / "text"
    if img_dir.is_dir():
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in _IMAGE_EXT:
                continue
            txt = (txt_dir / f"{img.stem}.txt") if txt_dir.is_dir() else img.with_suffix(".txt")
            if txt.exists():
                pairs.append((img, txt))
            else:
                logger.warning("no transcript for %s (expected %s)", img.name, txt)
    else:
        for img in sorted(data_dir.rglob("*")):
            if img.suffix.lower() in _IMAGE_EXT:
                txt = img.with_suffix(".txt")
                if txt.exists():
                    pairs.append((img, txt))
    return pairs


def char_coverage(texts: list[str], min_count: int) -> None:
    """Log per-character counts and flag characters seen fewer than `min_count` times.

    The model only learns a glyph it has seen often enough; a rare character
    (a diacritic, an honorific ligature, a digit) is a sign of which lines to
    annotate next.
    """
    counts = Counter(ch for t in texts for ch in t if not ch.isspace())
    rare = sorted((n, ch) for ch, n in counts.items() if n < min_count)
    logger.info(
        "character coverage: %d distinct, %d below %d instances", len(counts), len(rare), min_count
    )
    for n, ch in rare:
        shown = f"\u25cc{ch}" if unicodedata.combining(ch) else ch  # draw marks on a dotted circle
        name = unicodedata.name(ch, "?")
        logger.info("  %4d  %s  U+%04X %s", n, shown, ord(ch), name)


def check_data(data_dir: Path, strip: bool = False, min_char_count: int = 20) -> int:
    """Validate the dataset (no model load). Returns the number of usable pairs."""
    from PIL import Image  # local import keeps --help light

    if not data_dir.is_dir():
        logger.error("data directory not found: %s", data_dir)
        return 0
    pairs = find_pairs(data_dir)
    empty = 0
    bad = 0
    texts: list[str] = []
    for img, txt in pairs:
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception as exc:
            logger.warning("unreadable image %s: %s", img.name, exc)
            bad += 1
        text = load_target(txt, strip)
        if text:
            texts.append(text)
        else:
            empty += 1
    logger.info("found %d pairs (%d empty transcripts, %d unreadable images)", len(pairs), empty, bad)
    if empty:
        logger.warning("%d transcript(s) are still empty — transcribe them before training", empty)
    if len(pairs) < 200:
        logger.warning("only %d pairs; the plan targets >=800-1000 lines for a reliable model", len(pairs))
    char_coverage(texts, min_char_count)
    return len(pairs)


# --- Per-sample preparation -------------------------------------------------


def _chat_text(processor: Any, prompt_text: str) -> str:
    """Render the multimodal chat template so image placeholder tokens are present.

    Qwen2.5-VL's processor only expands `<|image_pad|>` into vision tokens when the
    text came from the chat template. Passing *raw* text together with `images=`
    produces `pixel_values`/`image_grid_thw` but ZERO image tokens in `input_ids`,
    which makes `get_placeholder_mask` raise:
        ValueError: Image features and image tokens do not match, tokens: 0, features: N
    """
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_sample(
    image: Any, target: str, processor: Any, prompt_text: str, eos: str
) -> dict[str, Any]:
    """Build one processed training example with prompt tokens masked out of the loss.

    The prompt is chat-templated (image tokens present), and the prompt-only pass
    uses the SAME image so the vision-token expansion matches and the prefix can be
    masked reliably in `labels`.

    Raises ValueError when the image collapsed to zero tiles / zero image tokens so
    the caller can skip the sample instead of crashing the model forward.
    """
    chat_prompt = _chat_text(processor, prompt_text)
    full = processor(text=[chat_prompt + target + eos], images=[image], return_tensors="pt")
    prompt = processor(text=[chat_prompt], images=[image], return_tensors="pt")

    # Guardrail: text must carry image placeholder tokens, and vision features must exist.
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_tokens = int((full["input_ids"] == image_pad_id).sum())
    if "pixel_values" not in full or "image_grid_thw" not in full:
        raise ValueError("processor returned no pixel_values/image_grid_thw")
    if n_tokens == 0 or full["pixel_values"].shape[0] == 0:
        raise ValueError(f"image collapsed to 0 tiles (image tokens={n_tokens})")

    prompt_len = prompt["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100

    batch = {
        k: full[k]
        for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        if k in full
    }
    batch["labels"] = labels
    return batch


# --- Training ---------------------------------------------------------------


def _load_model(model_id: str, device: Any, torch: Any) -> Any:
    """Load the Qwen2.5-VL (or Qwen2-VL) model in bf16 on `device`."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Model
    except Exception:
        from transformers import Qwen2VLForConditionalGeneration as _Model

    logger.info("loading base model %s ...", model_id)
    model = _Model.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.to(device)
    return model


def train(args: argparse.Namespace) -> int:
    import torch  # lazy: --check-data / --help must not need training deps
    from transformers import AutoProcessor

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        logger.error("peft is required for LoRA training: pip install -U peft (%s)", exc)
        return 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompt = _PLAIN_PROMPT if args.strip_diacritics else _LINE_PROMPT
    if args.strip_diacritics:
        logger.info("training on diacritic-stripped targets")

    all_pairs = find_pairs(args.data)
    if not all_pairs:
        logger.error("no image/text pairs found under %s", os_path(args.data))
        return 1
    # Drop pairs with empty transcripts: training on them is a no-op that silently
    # produces an untrained model (loss stays 0.0).
    pairs = [(i, t) for (i, t) in all_pairs if t.read_text(encoding="utf-8").strip()]
    if not pairs:
        logger.error(
            "all %d pairs have EMPTY transcripts — annotate the lines first: "
            "run `python prefill_line_gt.py --data %s`, then correct them by hand",
            len(all_pairs), os_path(args.data),
        )
        return 1
    if args.smoke_test:
        pairs = pairs[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s | pairs: %d", device, len(pairs))

    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token or ""

    model = _load_model(args.model, device, torch)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(pairs) / args.accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    from PIL import Image

    global_step = 0
    skipped = 0
    optim.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(pairs)))
        random.shuffle(order)
        epoch_loss = 0.0
        t0 = time.monotonic()
        for i, idx in enumerate(order):
            img_path, txt_path = pairs[idx]
            target = load_target(txt_path, args.strip_diacritics)
            if not target:
                continue
            image = Image.open(img_path).convert("RGB")
            try:
                batch = prepare_sample(image, target, processor, prompt, eos)
            except ValueError as exc:
                # e.g. an image the processor collapsed to zero tiles: skip, don't crash.
                skipped += 1
                logger.warning("skipping %s: %s", img_path.name, exc)
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss
            (loss / args.accum).backward()
            epoch_loss += float(loss.detach()) * args.accum

            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    lr = sched.get_last_lr()[0]
                    logger.info(
                        "epoch %d step %d/%d | loss %.4f | lr %.2e",
                        epoch, global_step, total_steps, epoch_loss / (i + 1), lr,
                    )
                if args.max_steps and global_step >= args.max_steps:
                    logger.info("reached --max-steps %d; stopping early", args.max_steps)
                    break

            if args.smoke_test:
                break

        # Flush any remaining accumulated gradients at the epoch end.
        if (len(order) % args.accum) != 0 and not args.smoke_test:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

        logger.info(
            "epoch %d done in %.1fs (mean loss %.4f)", epoch, time.monotonic() - t0, epoch_loss / max(1, len(order))
        )
        if args.max_steps and global_step >= args.max_steps:
            break
        if args.smoke_test:
            break

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the adapter (reproducible) and the MERGED model (loadable by the engine).
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    logger.info("saved LoRA adapter -> %s", adapter_dir)

    logger.info("merging LoRA weights ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "pairs": len(pairs),
                "epochs": args.epochs,
                "accum": args.accum,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "strip_diacritics": args.strip_diacritics,
                "prompt": prompt,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("done. merged model + processor -> %s (skipped %d unusable sample(s))", out_dir, skipped)
    return 0


def os_path(p: Path) -> str:
    return str(p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Arabic line OCR (Qwen2.5-VL).")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base HF model id.")
    p.add_argument("--data", type=Path, default=Path("annotation"), help="Dataset dir (images/ + text/).")
    p.add_argument("--out", type=Path, default=Path("models/qwen-arabic-lora"), help="Output dir for merged model.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--accum", type=int, default=8, help="Gradient accumulation (effective batch size).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--gradient-checkpointing", action="store_true", help="Lower VRAM, slightly slower.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps (0 = no limit).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check-data", action="store_true", help="Validate the dataset and exit (no model load).")
    p.add_argument("--smoke-test", action="store_true", help="Run one training step on one sample and exit.")
    p.add_argument(
        "--strip-diacritics",
        action="store_true",
        help="Train on targets with harakat removed (fewer classes; for search-only output).",
    )
    p.add_argument("--min-char-count", type=int, default=20, help="--check-data: flag rarer characters.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.check_data:
        n = check_data(args.data, args.strip_diacritics, args.min_char_count)
        return 0 if n else 1
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
