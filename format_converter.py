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

        escaped = _html.escape(clean).replace("\n", "<br>")
        html = (
            f'<div style="font-family: Noto Naskh Arabic; '
            f'font-size: 9px; direction: rtl; '
            f'text-align: right;">{escaped}</div>'
        )
        margin = 20
        rect = pymupdf.Rect(
            margin, margin,
            page.rect.width - margin, page.rect.height - margin,
        )
        # opacity=0 (native param) makes text invisible but still searchable.
        page.insert_htmlbox(rect, html, opacity=0)

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
                f'text-align: right;">{escaped}</div>'
            )
            # opacity=0 (native param) makes text invisible but still searchable.
            page.insert_htmlbox(rect, html, opacity=0)

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

                # Create the page with the scan as the visible image.
                new_page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
                new_page.insert_image(new_page.rect, stream=img_bytes)

                if not model_boxes or not cv_boxes:
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

                for box in model_boxes:
                    bbox = box.get("bbox_2d")
                    text = box.get("text", "").strip()
                    if not bbox or len(bbox) != 4 or not text:
                        continue
                    # Model bbox center in rendered-pixel space.
                    mx0, my0, mx1, my1 = bbox
                    mcx = ((mx0 + mx1) / 2) * mscale_x
                    mcy = ((my0 + my1) / 2) * mscale_y

                    # Snap to nearest OpenCV component (pixel-accurate position).
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
                    box_height_pt = (y1 - y0) * scale_y
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
