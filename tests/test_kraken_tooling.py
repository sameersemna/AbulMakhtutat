"""Unit tests for the Kraken backend plumbing, tooling helpers and canonicalization.

Run:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_errors import character_category, load_manifest  # noqa: E402
from canonicalize_gt import (  # noqa: E402
    MAPPING,
    append_new_files,
    canonicalize,
    rewrite_manifest,
    strip_tashkeel,
)


class TestCanonicalize(unittest.TestCase):
    def test_expands_sallallahu_ligature(self) -> None:
        self.assertEqual(canonicalize("\ufdfa"), "صلى الله عليه وسلم")

    def test_expands_allaa_anjah_ligature(self) -> None:
        self.assertEqual(canonicalize("\ufd42"), "رضي الله عنها")

    def test_keeps_ornate_parentheses(self) -> None:
        self.assertEqual(canonicalize("﴿نص﴾"), "﴿نص﴾")

    def test_plain_text_unchanged(self) -> None:
        self.assertEqual(canonicalize("نص عادي"), "نص عادي")

    def test_mapping_entries_are_real(self) -> None:
        for ch, phrase in MAPPING.items():
            self.assertEqual(len(ch), 1)
            self.assertTrue(phrase.strip())
            self.assertNotIn(ch, phrase)


class TestStripTashkeel(unittest.TestCase):
    def test_removes_harakat_keeps_letters(self) -> None:
        self.assertEqual(strip_tashkeel("مُحَمَّد"), "محمد")

    def test_keeps_quranic_superscript_alef(self) -> None:
        self.assertEqual(strip_tashkeel("علىٰ الحمد"), "علىٰ الحمد")


class TestRewriteManifest(unittest.TestCase):
    def test_matches_by_basename_regardless_of_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "dst"
            (dst / "images").mkdir(parents=True)
            (dst / "images" / "a.png").write_bytes(b"x")
            manifest = Path(tmp) / "in.txt"
            manifest.write_text("annotation_nodiac/images/a.png\nmissing/b.png\n", encoding="utf-8")
            out = Path(tmp) / "out.txt"
            changed = rewrite_manifest(dst, manifest, out)
            self.assertEqual(changed, 1)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertTrue(lines[0].endswith("dst/images/a.png"))
            self.assertEqual(lines[1], "missing/b.png")


class TestAppendNewFiles(unittest.TestCase):
    def test_appends_only_unknown_files_to_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "dst"
            images = dst / "images"
            images.mkdir(parents=True)
            for name in ("a", "b", "c"):
                (images / f"{name}.png").write_bytes(b"x")
                (images / f"{name}.gt.txt").write_text("نص", encoding="utf-8")
            train = Path(tmp) / "train.txt"
            holdout = Path(tmp) / "holdout.txt"
            train.write_text(str(images / "a.png") + "\n", encoding="utf-8")
            holdout.write_text(str(images / "b.png") + "\n", encoding="utf-8")
            appended = append_new_files(dst, train, holdout)
            self.assertEqual(appended, 1)
            self.assertEqual([Path(line).name for line in train.read_text(encoding="utf-8").splitlines()],
                             ["a.png", "c.png"])
            self.assertEqual(holdout.read_text(encoding="utf-8").splitlines(), [str(images / "b.png")])

    def test_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "dst"
            images = dst / "images"
            images.mkdir(parents=True)
            (images / "a.png").write_bytes(b"x")
            (images / "a.gt.txt").write_text("نص", encoding="utf-8")
            train = Path(tmp) / "train.txt"
            holdout = Path(tmp) / "holdout.txt"
            train.write_text(str(images / "a.png") + "\n", encoding="utf-8")
            holdout.write_text("", encoding="utf-8")
            self.assertEqual(append_new_files(dst, train, holdout), 0)
            self.assertEqual(train.read_text(encoding="utf-8").splitlines(), [str(images / "a.png")])


class TestCharacterCategory(unittest.TestCase):
    def test_arabic_indic_digit(self) -> None:
        self.assertEqual(character_category("٣"), "digit")
        self.assertEqual(character_category("5"), "digit")

    def test_arabic_letter(self) -> None:
        self.assertEqual(character_category("ا"), "arabic letter")

    def test_honorific_ligature(self) -> None:
        self.assertEqual(character_category("\ufdfa"), "honorific/presentation")

    def test_ornate_paren(self) -> None:
        self.assertEqual(character_category("﴾"), "ornate paren")

    def test_space_and_empty(self) -> None:
        self.assertEqual(character_category(" "), "space")
        self.assertEqual(character_category(""), "empty")

    def test_quranic_mark(self) -> None:
        self.assertEqual(character_category("\u0670"), "quranic mark")


class TestManifestLoading(unittest.TestCase):
    def test_loads_existing_skips_missing_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            good = base / "a.png"
            good.write_bytes(b"x")
            manifest = base / "manifest.txt"
            manifest.write_text("\n".join(["# comment", str(good), str(base / "missing.png"), ""]),
                                encoding="utf-8")
            files = load_manifest(manifest)
            self.assertEqual(files, [good])


class TestLineSegmentation(unittest.TestCase):
    def test_bbox_covers_whole_image(self) -> None:
        from PIL import Image

        from kraken_engine import line_segmentation

        im = Image.new("RGB", (40, 25))
        seg = line_segmentation(im)
        self.assertEqual(seg.type, "bbox")
        self.assertEqual(len(seg.lines), 1)
        self.assertEqual(seg.lines[0].bbox, (0, 0, 40, 25))


class TestEngineFactory(unittest.TestCase):
    def test_kraken_missing_model_falls_back_to_ollama(self) -> None:
        from config import get_config
        from hf_vlm_engine import build_ocr_engine
        from ocr_engine import OCREngine

        cfg = get_config()
        cfg.vision.backend = "kraken"
        cfg.vision.kraken_model = "/nonexistent/model.safetensors"
        engine = build_ocr_engine(cfg)
        self.assertIsInstance(engine, OCREngine)


if __name__ == "__main__":
    unittest.main()
