"""Multi-format output generation.

Converts the canonical Markdown into `.md`, `.html`, `.epub`, `.mobi`, and
`.pdf` within the document's mapped output directory. Uses Pandoc for HTML/EPUB,
WeasyPrint for PDF (best RTL support), and Calibre's `ebook-convert` for MOBI
(with a graceful skip if Calibre is unavailable).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from config import Config, get_config
from models import DocumentResult

logger = logging.getLogger(__name__)

_RTL_CSS = """
@page {
    size: A4;
    margin: 2cm;
    @bottom-center { content: counter(page); }
}
html, body {
    direction: rtl;
    text-align: right;
    font-family: 'Amiri', 'Traditional Arabic', 'Scheherazade New', serif;
    line-height: 1.9;
}
.page-anchor {
    page-break-before: always;
    break-before: page;
}
h1, h2, h3, h4, h5, h6 { direction: rtl; text-align: right; }
p, li, blockquote { direction: rtl; text-align: right; }
"""


class FormatConverter:
    """Compiles canonical Markdown into all target formats."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

    def _write_markdown(self, result: DocumentResult, markdown: str) -> Path:
        path = result.output_dir / f"{result.source_path.stem}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def _write_html(self, result: DocumentResult, markdown: str) -> Path:
        path = result.output_dir / f"{result.source_path.stem}.html"
        css = _RTL_CSS.replace("'Amiri', 'Traditional Arabic', 'Scheherazade New', serif", self.config.pipeline.font_family)
        html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{result.source_path.stem}</title>
<style>
{css}
</style>
</head>
<body>
{self._markdown_to_html(markdown)}
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert Markdown body to HTML via pandoc (fragment)."""
        try:
            proc = subprocess.run(
                ["pandoc", "-f", "markdown+raw_html", "-t", "html"],
                input=markdown,
                capture_output=True,
                text=True,
                check=True,
            )
            return proc.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("pandoc unavailable for HTML body, using raw markdown: %s", exc)
            return f"<pre>{markdown}</pre>"

    def _write_epub(self, result: DocumentResult, markdown: str) -> Path | None:
        path = result.output_dir / f"{result.source_path.stem}.epub"
        if not shutil.which("pandoc"):
            logger.warning("pandoc not found; skipping EPUB for %s", result.rel_path)
            return None
        try:
            subprocess.run(
                [
                    "pandoc",
                    "-f", "markdown+raw_html",
                    "-t", "epub3",
                    "--metadata", f"title={result.source_path.stem}",
                    "--metadata", "lang=ar",
                    "--css", self._write_css_file(result),
                    "-o", str(path),
                ],
                input=markdown,
                capture_output=True,
                text=True,
                check=True,
            )
            return path
        except subprocess.CalledProcessError as exc:
            logger.error("EPUB generation failed for %s: %s", result.rel_path, exc.stderr)
            return None

    def _write_css_file(self, result: DocumentResult) -> str:
        css_path = result.output_dir / "style.css"
        css_path.write_text(_RTL_CSS, encoding="utf-8")
        return str(css_path)

    def _write_pdf(self, result: DocumentResult, markdown: str) -> Path | None:
        path = result.output_dir / f"{result.source_path.stem}.pdf"
        try:
            from weasyprint import HTML  # type: ignore

            html = self._markdown_to_html(markdown)
            css = _RTL_CSS.replace("'Amiri', 'Traditional Arabic', 'Scheherazade New', serif", self.config.pipeline.font_family)
            full = f"<html lang='ar' dir='rtl'><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
            HTML(string=full).write_pdf(str(path))
            return path
        except Exception as exc:
            logger.error("PDF generation failed for %s: %s", result.rel_path, exc)
            return None

    def _ebook_convert_bin(self) -> str | None:
        """Resolve the Calibre ebook-convert binary path."""
        configured = self.config.pipeline.ebook_convert
        if configured:
            p = Path(configured).expanduser()
            if p.exists():
                return str(p)
            logger.warning("Configured ebook-convert not found at %s", configured)
        return shutil.which("ebook-convert")

    def _write_mobi(self, result: DocumentResult) -> Path | None:
        path = result.output_dir / f"{result.source_path.stem}.mobi"
        epub = result.output_dir / f"{result.source_path.stem}.epub"
        ebook_convert = self._ebook_convert_bin()
        if not ebook_convert:
            logger.warning("Calibre `ebook-convert` not found; skipping MOBI for %s", result.rel_path)
            return None
        if not epub.exists():
            logger.warning("EPUB missing; cannot build MOBI for %s", result.rel_path)
            return None
        try:
            subprocess.run(
                [ebook_convert, str(epub), str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return path
        except subprocess.CalledProcessError as exc:
            logger.error("MOBI generation failed for %s: %s", result.rel_path, exc.stderr)
            return None

    def _write_overlay_pdf(self, result: DocumentResult) -> Path | None:
        """Generate a searchable PDF: original scan + invisible OCR text layer.

        Renders each source page as a full-page image (the visible layer), then
        overlays the OCR text as invisible text so the PDF looks identical to
        the scan but is searchable/copyable.

        Uses `insert_htmlbox` (MuPDF's HTML engine with HarfBuzz shaping) with
        `opacity: 0` — this is required for Arabic, because `insert_text` /
        `insert_textbox` do NOT perform Arabic contextual shaping and produce
        unsearchable `?` glyphs. The HTML path shapes Arabic correctly.

        Two modes:
          - "page": full-page text block per page (approximate alignment).
          - "word": line-aligned placement using OpenCV-detected text-line boxes
            (closer spatial alignment to the scan).
        """
        import html as _html
        import re

        import pymupdf

        from preprocess import detect_line_boxes

        mode = self.config.pipeline.overlay_mode
        if mode in ("off", "bbox"):
            # "bbox" mode is handled separately (async, needs the engine).
            return None

        font_path = self.config.pipeline.overlay_font
        if not Path(font_path).exists():
            logger.warning("Overlay font not found at %s; skipping overlay PDF for %s", font_path, result.rel_path)
            return None

        out_path = result.output_dir / f"{result.source_path.stem}_overlay.pdf"
        try:
            src = pymupdf.open(str(result.source_path))
            out = pymupdf.open()  # new empty PDF

            for page in result.pages:
                page_num = page.page_number
                if page_num < 1 or page_num > src.page_count:
                    continue

                # 1. Render the source page as the visible image layer.
                src_page = src[page_num - 1]
                zoom = self.config.pipeline.dpi / 72.0
                pix = src_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                img_bytes = pix.tobytes("png")

                # 2. Create a new page with the same dimensions as the source.
                new_page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
                new_page.insert_image(new_page.rect, stream=img_bytes)

                # 3. Overlay invisible OCR text (opacity 0, still searchable).
                text = page.text.strip()
                if not text or page.error:
                    continue

                # Strip Markdown syntax so the searchable layer is clean text.
                clean = self._strip_markdown(text)
                if not clean.strip():
                    continue

                if mode == "word":
                    self._overlay_word_aligned(new_page, clean, img_bytes, _html)
                else:
                    self._overlay_full_page(new_page, clean, _html)

            out.save(str(out_path))
            out.close()
            src.close()
            return out_path
        except Exception as exc:
            logger.error("Overlay PDF generation failed for %s: %s", result.rel_path, exc)
            return None

    def _overlay_full_page(self, page, clean: str, _html) -> None:
        """Overlay the full page text as a single invisible block."""
        import pymupdf

        debug = self.config.pipeline.overlay_debug
        color_style = "color: red;" if debug else ""
        escaped = _html.escape(clean).replace("\n", "<br>")
        html = (
            f'<div style="font-family: Noto Naskh Arabic; '
            f'font-size: 9px; direction: rtl; '
            f'text-align: right; {color_style}">{escaped}</div>'
        )
        margin = 20
        rect = pymupdf.Rect(
            margin, margin,
            page.rect.width - margin, page.rect.height - margin,
        )
        # opacity=0 (native param) makes text invisible but still searchable.
        page.insert_htmlbox(rect, html, opacity=0 if not debug else 1)

    def _overlay_word_aligned(self, page, clean: str, img_bytes: bytes, _html) -> None:
        """Overlay text line-by-line, aligned to OpenCV-detected text-line boxes."""
        import pymupdf

        from preprocess import detect_line_boxes

        # Detect text-line boxes from the rendered page image.
        line_boxes = detect_line_boxes(img_bytes)
        lines = [l for l in clean.splitlines() if l.strip()]

        if not line_boxes or not lines:
            # Fall back to full-page if detection failed.
            self._overlay_full_page(page, clean, _html)
            return

        # Map OCR lines onto detected line boxes (top-to-bottom).
        # If counts differ, fall back to full-page (safer than misalignment).
        if len(line_boxes) < len(lines):
            self._overlay_full_page(page, clean, _html)
            return

        # The rendered image is at `dpi` zoom; convert pixel coords to PDF points.
        zoom = self.config.pipeline.dpi / 72.0
        debug = self.config.pipeline.overlay_debug
        color_style = "color: red;" if debug else ""
        for i, line in enumerate(lines):
            if i >= len(line_boxes):
                break
            x0, y0, x1, y1 = line_boxes[i]
            # Convert pixel coords (at dpi) to PDF points.
            rect = pymupdf.Rect(
                x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom,
            )
            escaped = _html.escape(line)
            html = (
                f'<div style="font-family: Noto Naskh Arabic; '
                f'font-size: 9px; direction: rtl; '
                f'text-align: right; {color_style}">{escaped}</div>'
            )
            # opacity=0 (native param) makes text invisible but still searchable.
            page.insert_htmlbox(rect, html, opacity=0 if not debug else 1)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove Markdown syntax (headings, lists, quotes, emphasis) for the
        searchable text layer, leaving clean readable text."""
        import re

        lines = []
        for line in text.splitlines():
            # Strip heading markers (#, ##, ...).
            line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
            # Strip list markers (-, *, 1.).
            line = re.sub(r"^\s{0,3}([-*+]|\d+[.)])\s+", "", line)
            # Strip blockquote markers (>).
            line = re.sub(r"^\s{0,3}>\s?", "", line)
            # Strip inline emphasis/strong markers.
            line = line.replace("**", "").replace("__", "").replace("*", "").replace("`", "")
            lines.append(line)
        return "\n".join(lines)

    def convert(self, result: DocumentResult, markdown: str) -> dict[str, Path | None]:
        """Generate all output formats. Returns a map of format -> path."""
        outputs: dict[str, Path | None] = {}
        outputs["md"] = self._write_markdown(result, markdown)
        outputs["html"] = self._write_html(result, markdown)
        outputs["epub"] = self._write_epub(result, markdown)
        outputs["pdf"] = self._write_pdf(result, markdown)
        outputs["mobi"] = self._write_mobi(result)
        outputs["overlay"] = self._write_overlay_pdf(result)
        return outputs

    async def write_overlay_bbox(self, result: DocumentResult, engine) -> Path | None:
        """Generate a searchable PDF with model-precise bounding-box alignment.

        Uses the vision model's visual grounding (`bbox_2d` coordinates) to place
        each OCR text line exactly where the model detected it, then renders it
        as an invisible (opacity=0) but searchable text layer over the scan.

        This is the most accurate overlay mode: position AND font size are
        derived from the model's own bounding boxes, so selection highlights
        match the scanned text closely.
        """
        import html as _html
        import io as _io

        import pymupdf
        from PIL import Image

        from preprocess import detect_words_reading_order

        mode = self.config.pipeline.overlay_mode
        if mode != "bbox":
            return None

        font_path = self.config.pipeline.overlay_font
        if not Path(font_path).exists():
            logger.warning("Overlay font not found at %s; skipping bbox overlay", font_path)
            return None

        out_path = result.output_dir / f"{result.source_path.stem}_overlay.pdf"
        try:
            src = pymupdf.open(str(result.source_path))
            out = pymupdf.open()

            # Load the Arabic font once for exact metric-based sizing.
            font = pymupdf.Font(fontfile=font_path)
            line_height_ratio = font.ascender - font.descender

            for page in result.pages:
                page_num = page.page_number
                if page_num < 1 or page_num > src.page_count:
                    continue

                src_page = src[page_num - 1]
                zoom = self.config.pipeline.dpi / 72.0
                pix = src_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                img_bytes = pix.tobytes("png")

                # Get model word text + approximate bbox, and OpenCV pixel-accurate
                # word components. The model's text is accurate and its word count
                # is correct; OpenCV's boxes are pixel-accurate but over-segment
                # (diacritics/dots become separate components). We snap each model
                # word to the nearest OpenCV component for precise placement.
                model_boxes = await engine.ocr_page_with_boxes(img_bytes)
                cv_boxes = detect_words_reading_order(img_bytes)

                # Filter out oversized OpenCV components. On title/cover pages,
                # large decorative text can merge into a single full-page box
                # (e.g. (0,0,2480,3114)); snapping model words to it produces
                # huge font sizes that overflow the page. A real word component
                # is small relative to the page.
                if cv_boxes:
                    pw, ph = pix.width, pix.height
                    cv_boxes = [
                        b for b in cv_boxes
                        if (b[2] - b[0]) <= 0.5 * pw
                        and (b[3] - b[1]) <= 0.5 * ph
                        and (b[2] - b[0]) * (b[3] - b[1]) <= 0.25 * pw * ph
                    ]

                # Create the page with the scan as the visible image.
                new_page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
                new_page.insert_image(new_page.rect, stream=img_bytes)

                if not model_boxes:
                    continue

                # Scale: rendered image pixels -> PDF points.
                scale_x = src_page.rect.width / pix.width
                scale_y = src_page.rect.height / pix.height

                # Determine the downscaled image dims (model bbox space).
                max_dim = self.config.ollama.max_image_dim
                im = Image.open(_io.BytesIO(img_bytes))
                im.thumbnail((max_dim, max_dim))
                ds_w, ds_h = im.size
                # Model bbox (downscaled px) -> rendered px.
                mscale_x = pix.width / ds_w
                mscale_y = pix.height / ds_h

                # If OpenCV detection is too poor (title/cover pages with large
                # decorative text), fall back to the model's own bbox directly.
                # Snapping to a handful of tiny fragments would shrink every word
                # to a few pixels. Adequate detection = enough boxes to cover the
                # model's word count.
                use_model_bbox = (
                    not cv_boxes
                    or len(cv_boxes) < 5
                    or len(cv_boxes) < 0.3 * len(model_boxes)
                )

                for box in model_boxes:
                    bbox = box.get("bbox_2d")
                    text = box.get("text", "").strip()
                    if not bbox or len(bbox) != 4 or not text:
                        continue

                    if use_model_bbox:
                        # Use the model's own bbox (downscaled px -> PDF points).
                        mx0, my0, mx1, my1 = bbox
                        rect = pymupdf.Rect(
                            mx0 * mscale_x * scale_x,
                            my0 * mscale_y * scale_y,
                            mx1 * mscale_x * scale_x,
                            my1 * mscale_y * scale_y,
                        )
                    else:
                        # Snap to nearest OpenCV component (pixel-accurate position).
                        mcx = ((bbox[0] + bbox[2]) / 2) * mscale_x
                        mcy = ((bbox[1] + bbox[3]) / 2) * mscale_y
                        best = None
                        best_dist = float("inf")
                        for (cx0, cy0, cx1, cy1) in cv_boxes:
                            ccx = (cx0 + cx1) / 2
                            ccy = (cy0 + cy1) / 2
                            d = (ccx - mcx) ** 2 + (ccy - mcy) ** 2
                            if d < best_dist:
                                best_dist = d
                                best = (cx0, cy0, cx1, cy1)
                        if best is None:
                            continue
                        x0, y0, x1, y1 = best
                        rect = pymupdf.Rect(
                            x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y,
                        )

                    box_height_pt = rect.height
                    font_size = max(4.0, box_height_pt / line_height_ratio)
                    escaped = _html.escape(text)
                    color_style = "color: red;" if self.config.pipeline.overlay_debug else ""
                    html = (
                        f'<div style="font-family: Noto Naskh Arabic; '
                        f'font-size: {font_size:.2f}px; direction: rtl; '
                        f'text-align: right; {color_style}">{escaped}</div>'
                    )
                    new_page.insert_htmlbox(rect, html, opacity=0 if not self.config.pipeline.overlay_debug else 1)

            out.save(str(out_path))
            out.close()
            src.close()
            return out_path
        except Exception as exc:
            logger.error("BBox overlay generation failed for %s: %s", result.rel_path, exc)
            return None

    async def write_overlay_line(self, result: DocumentResult, engine, thermal=None) -> Path | None:
        """Generate a searchable PDF with LINE-level alignment.

        Uses the model's accurate line text (complete content — no missing words,
        unlike word-level which under-segments Arabic) placed at detected
        line boxes (pixel-accurate positions). Each line's full text is sized to
        fill the line box height via line-height:1.0.

        This is the most accurate achievable alignment: complete text, correct
        vertical position, and font size that fills the scan's text lines.

        Args:
            thermal: Optional ThermalGuard to gate GPU inference (Surya) so the
                GPU can work hard while throttling safely via queueing.
        """
        import html as _html
        import io as _io

        import pymupdf
        from PIL import Image

        from preprocess import detect_line_boxes

        mode = self.config.pipeline.overlay_mode
        if mode != "line":
            return None

        font_path = self.config.pipeline.overlay_font
        if not Path(font_path).exists():
            logger.warning("Overlay font not found at %s; skipping line overlay", font_path)
            return None

        out_path = result.output_dir / f"{result.source_path.stem}_overlay.pdf"
        try:
            src = pymupdf.open(str(result.source_path))
            out = pymupdf.open()

            font = pymupdf.Font(fontfile=font_path)
            line_height_ratio = font.ascender - font.descender

            for page in result.pages:
                page_num = page.page_number
                if page_num < 1 or page_num > src.page_count:
                    continue

                src_page = src[page_num - 1]
                zoom = self.config.pipeline.dpi / 72.0
                pix = src_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                raw_bytes = pix.tobytes("png")

                # Apply the SAME preprocessing the OCR text was produced from, so
                # the background image and ink-height calibration are consistent
                # with the OCR text. Without this, the geometry (raw scan) and the
                # text (preprocessed) diverge and word-fitting across line boxes
                # produces misaligned overlays.
                from preprocess import preprocess
                img_bytes = preprocess(raw_bytes, self.config)

                # Model line text (complete content) + detection line boxes
                # (pixel-accurate positions). Snap each model line to the nearest
                # detected line box for precise placement.
                line_boxes = await engine.ocr_page_with_line_boxes(img_bytes)

                # Prefer a purpose-built detection engine (Surya/EasyOCR) for
                # pixel-accurate line boxes; fall back to OpenCV if unavailable.
                # IMPORTANT: run detection on the RAW image, not the preprocessed
                # one. Binarization strips the subtle gradients Surya relies on to
                # find text lines (33 lines on raw vs 5 on binarized for page 33),
                # so detection on the preprocessed image drops most lines. Since
                # deskew preserves image dimensions, raw line boxes map 1:1 onto
                # the preprocessed background.
                cv_lines = []
                detector = self.config.pipeline.overlay_detector
                device = self.config.pipeline.overlay_device
                if detector in ("surya", "easyocr"):
                    try:
                        im = Image.open(_io.BytesIO(raw_bytes))
                        # Gate GPU inference through the thermal guard so the GPU
                        # can work hard while throttling safely (queueing, not CPU).
                        if device == "cuda" and thermal is not None:
                            await thermal.wait_if_hot()
                        if detector == "surya":
                            from surya_detection import detect_line_boxes_surya
                            cv_lines = detect_line_boxes_surya(im, device=device)
                        else:
                            from easyocr_detection import detect_line_boxes_easyocr
                            cv_lines = detect_line_boxes_easyocr(im, device=device)
                    except Exception as exc:
                        logger.warning("%s detection unavailable (%s); using OpenCV", detector, exc)
                        cv_lines = []
                if not cv_lines:
                    cv_lines = detect_line_boxes(raw_bytes)

                # Filter detection line boxes: drop boxes that are too tall to be
                # a real text line. On title/cover pages, decorative text can
                # merge into a giant box spanning most of the page (e.g.
                # (0,258,2431,3165) on page 2, or (0,0,2480,3114) on page 1). A
                # legitimate Arabic text line is only ~85-96px tall at 300 DPI,
                # so any box taller than half the page is an artifact. Width
                # alone must NOT trigger removal — justified Arabic lines span
                # nearly the full width.
                if cv_lines:
                    ph = pix.height
                    cv_lines = [
                        b for b in cv_lines
                        if (b[3] - b[1]) <= 0.5 * ph
                    ]

                new_page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
                new_page.insert_image(new_page.rect, stream=img_bytes)

                # Use the COMPLETE main OCR text for this page (from the cached
                # result), not the LLM's line-box text. The LLM's line detection
                # under-segments Arabic (e.g. 6 boxes vs 33 visual lines on page
                # 33), dropping most lines. The main OCR text is complete and
                # correct; we split it across the detected line boxes by fitting
                # words to each box's width.
                full_text = self._strip_markdown(page.text).strip()
                if not full_text:
                    continue

                # Downscaled image dims (model bbox space).
                max_dim = self.config.ollama.max_image_dim
                im = Image.open(_io.BytesIO(img_bytes))
                im.thumbnail((max_dim, max_dim))
                ds_w, ds_h = im.size

                # If detection failed (title/cover page with only a giant
                # decorative box), fall back to the model's own line boxes.
                if not cv_lines:
                    cv_lines = [
                        (b[0] * pix.width / ds_w, b[1] * pix.height / ds_h,
                         b[2] * pix.width / ds_w, b[3] * pix.height / ds_h)
                        for b in line_boxes if b.get("bbox_2d")
                    ]

                # Split the full text into words, preserving order.
                words = full_text.split()
                word_idx = 0

                # Load the scan image as grayscale once for ink-height measurement.
                import numpy as np
                scan_arr = np.array(Image.open(_io.BytesIO(img_bytes)).convert("L"))

                for (cx0, cy0, cx1, cy1) in cv_lines:
                    if word_idx >= len(words):
                        break
                    # Convert detection line box (rendered px) -> PDF points.
                    rect = pymupdf.Rect(
                        cx0 / pix.width * src_page.rect.width,
                        cy0 / pix.height * src_page.rect.height,
                        cx1 / pix.width * src_page.rect.width,
                        cy1 / pix.height * src_page.rect.height,
                    )
                    # Measure the ACTUAL scan ink height within this line box.
                    # Measure the scan's BASE LETTER height (excluding diacritics).
                    # The scan ink includes diacritics (harakat) above/below the
                    # base letters; matching the full ink height makes the base
                    # text too big. We use dark-pixel DENSITY per row: base
                    # letters are dense, diacritics are sparse. We match the
                    # base-letter height on BOTH the scan and the rendered text.
                    band = scan_arr[cy0:cy1, cx0:cx1]
                    dark = band < 100
                    row_density = dark.sum(axis=1)
                    max_density = row_density.max()
                    if max_density > 0:
                        base_rows = np.where(row_density > 0.05 * max_density)[0]
                        if len(base_rows) >= 2:
                            scan_base_h_px = base_rows.max() - base_rows.min()
                            scan_base_h_pt = scan_base_h_px / (self.config.pipeline.dpi / 72.0)
                        else:
                            scan_base_h_px = rect.height * (self.config.pipeline.dpi / 72.0)
                            scan_base_h_pt = rect.height
                    else:
                        scan_base_h_px = rect.height * (self.config.pipeline.dpi / 72.0)
                        scan_base_h_pt = rect.height

                    # Fit as many words as fit in the box width (using a test
                    # font size based on the scan base-letter height).
                    test_font_size = max(4.0, scan_base_h_pt)
                    line_words = []
                    line_width = 0.0
                    for w in words[word_idx:]:
                        ww = font.text_length(w, fontsize=test_font_size)
                        # Account for a space between words.
                        gap = font.text_length(" ", fontsize=test_font_size) if line_words else 0.0
                        if line_width + gap + ww <= rect.width * 0.98:
                            line_words.append(w)
                            line_width += gap + ww
                        else:
                            break
                    if not line_words:
                        # Single word too wide — scale font down to fit the box.
                        line_words = [words[word_idx]]
                        ww = font.text_length(line_words[0], fontsize=test_font_size)
                        if ww > rect.width:
                            test_font_size = max(4.0, test_font_size * rect.width / ww)
                    word_idx += len(line_words)
                    line_text = " ".join(line_words)

                    # CLOSED-LOOP calibration: render at the test size in a
                    # SEPARATE temp document, measure the rendered BASE LETTER
                    # height (dense rows, excluding diacritics), then scale to
                    # match the scan's base-letter height. Using a separate
                    # document avoids corrupting the main output document.
                    test_doc = pymupdf.open()
                    test_page = test_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
                    test_rect = pymupdf.Rect(
                        rect.x0, rect.y0, rect.x1, rect.y0 + test_font_size,
                    )
                    test_html = (
                        f'<div style="font-family: Noto Naskh Arabic; '
                        f'font-size: {test_font_size:.2f}px; line-height: 1.0; '
                        f'direction: rtl; text-align: right; color: red;">'
                        f'{_html.escape(line_text)}</div>'
                    )
                    test_page.insert_htmlbox(test_rect, test_html, opacity=1)
                    test_pix = test_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    test_arr = np.frombuffer(test_pix.samples, dtype=np.uint8).reshape(
                        test_pix.height, test_pix.width, test_pix.n
                    )
                    test_red = (test_arr[:, :, 0] > 150) & (test_arr[:, :, 1] < 100) & (test_arr[:, :, 2] < 100)
                    test_doc.close()
                    if test_red.any():
                        test_density = test_red.sum(axis=1)
                        test_max = test_density.max()
                        if test_max > 0:
                            test_base_rows = np.where(test_density > 0.05 * test_max)[0]
                            if len(test_base_rows) >= 2:
                                rendered_base_h_px = test_base_rows.max() - test_base_rows.min()
                                # Scale font size so rendered base height matches scan base height.
                                font_size = max(4.0, test_font_size * (scan_base_h_px / rendered_base_h_px))
                            else:
                                font_size = test_font_size
                        else:
                            font_size = test_font_size
                    else:
                        font_size = test_font_size

                    # Set the box height to the font size so the glyphs fill the
                    # scan ink height and text stays on ONE line (no invisible
                    # wrapped second line that breaks selection).
                    render_rect = pymupdf.Rect(
                        rect.x0, rect.y0, rect.x1, rect.y0 + font_size,
                    )
                    escaped = _html.escape(line_text)
                    color_style = "color: red;" if self.config.pipeline.overlay_debug else ""
                    html = (
                        f'<div style="font-family: Noto Naskh Arabic; '
                        f'font-size: {font_size:.2f}px; line-height: 1.0; '
                        f'direction: rtl; text-align: right; {color_style}">{escaped}</div>'
                    )
                    new_page.insert_htmlbox(render_rect, html, opacity=0 if not self.config.pipeline.overlay_debug else 1)

            out.save(str(out_path))
            out.close()
            src.close()
            return out_path
        except Exception as exc:
            logger.error("Line overlay generation failed for %s: %s", result.rel_path, exc)
            return None
