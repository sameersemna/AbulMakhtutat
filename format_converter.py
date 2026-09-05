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

        This is the "page" mode (full-page text block per page). Word-aligned
        placement requires bounding boxes from the OCR step (future work).
        """
        import html as _html
        import re

        import pymupdf

        mode = self.config.pipeline.overlay_mode
        if mode == "off":
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

                # Escape for HTML and preserve line breaks.
                escaped = _html.escape(clean).replace("\n", "<br>")
                html = (
                    f'<div style="font-family: Noto Naskh Arabic; '
                    f'font-size: 9px; opacity: 0; direction: rtl; '
                    f'text-align: right;">{escaped}</div>'
                )
                margin = 20
                rect = pymupdf.Rect(
                    margin, margin,
                    new_page.rect.width - margin, new_page.rect.height - margin,
                )
                new_page.insert_htmlbox(rect, html)

            out.save(str(out_path))
            out.close()
            src.close()
            return out_path
        except Exception as exc:
            logger.error("Overlay PDF generation failed for %s: %s", result.rel_path, exc)
            return None

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
