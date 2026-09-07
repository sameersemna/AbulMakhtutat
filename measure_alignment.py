"""Measure the spatial alignment accuracy of an overlay text layer vs a scan.

This is an INDEPENDENT verification tool for our main task: how accurately does
the (red, debug) overlay text sit on top of the original scan's text?

For each page it renders the overlay PDF and the original scan at the SAME DPI,
then compares the red overlay-text pixels against the dark scan-text pixels.

Metrics (per page + overall):
  - IoU        : |red ∩ dark| / |red ∪ dark|   (higher = better)
  - Precision  : |red ∩ dark| / |red|          (fraction of overlay text on scan text)
  - Recall     : |red ∩ dark| / |dark|          (fraction of scan text covered)
  - F1         : harmonic mean of precision & recall
  - Mean dist  : mean distance (px) from each red pixel to nearest dark pixel
  - Median dist: median distance (px) from each red pixel to nearest dark pixel

Interpretation (mirrors hypothesis/pdf-text-quality's iou_weighted):
  - IoU > 0.7  : good alignment
  - IoU 0.5-0.7: fair
  - IoU < 0.5  : poor

Usage:
    .venv/bin/python measure_alignment.py --overlay output/.../1_overlay.pdf \
        --scan input/rabee/majmua/1.pdf --dpi 150 --pages 1-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image


def _render_page(pdf_path: Path, page_index: int, dpi: int) -> np.ndarray:
    """Render a PDF page to a grayscale numpy array at the given DPI."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[page_index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if arr.shape[2] == 4:  # RGBA
        arr = arr[:, :, :3]
    return arr


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    """Red overlay text pixels (R high, G/B low)."""
    return (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)


def _dark_mask(rgb: np.ndarray) -> np.ndarray:
    """Dark scan text pixels (low luminance)."""
    gray = rgb.mean(axis=2)
    return gray < 100


def _distance_transform(binary: np.ndarray) -> np.ndarray:
    """Euclidean distance from each pixel to nearest True pixel (in px)."""
    import cv2

    # cv2.distanceTransform expects 0 = background, 255 = foreground.
    fg = (binary * 255).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    return dist


def _text_line_bands(mask: np.ndarray, min_row_dark: int = 20) -> list[tuple[int, int]]:
    """Find contiguous horizontal bands of rows containing text pixels.

    Returns a list of (y_start, y_end) bands. Used to compare the vertical
    position of overlay text lines vs scan text lines.
    """
    row_dark = mask.sum(axis=1)
    bands = []
    in_band = False
    for y in range(len(row_dark)):
        if row_dark[y] > min_row_dark:
            if not in_band:
                start = y
                in_band = True
        else:
            if in_band:
                bands.append((start, y))
                in_band = False
    if in_band:
        bands.append((start, len(row_dark)))
    return bands


def _line_position_accuracy(overlay_rgb: np.ndarray, scan_rgb: np.ndarray) -> dict:
    """Measure how well overlay text lines align vertically with scan text lines.

    This is the metric that matters for a SEARCHABLE PDF: does each overlay text
    line sit at the same vertical position as a scan text line? It does NOT
    require glyph-shape overlap (which is impossible without the original font).

    Returns:
      - overlay_lines: number of overlay text line bands
      - scan_lines: number of scan text line bands
      - matched_lines: overlay lines whose center falls within a scan line band
      - line_match_ratio: matched_lines / max(overlay_lines, 1)  (higher = better)
      - mean_center_offset_px: mean |overlay line center - nearest scan line center|
    """
    red = _red_mask(overlay_rgb)
    dark = _dark_mask(scan_rgb)

    overlay_bands = _text_line_bands(red)
    scan_bands = _text_line_bands(dark)

    if not overlay_bands or not scan_bands:
        return {
            "overlay_lines": len(overlay_bands),
            "scan_lines": len(scan_bands),
            "matched_lines": 0,
            "line_match_ratio": 0.0,
            "mean_center_offset_px": float("nan"),
        }

    # For each overlay line, find the nearest scan line by center distance.
    matched = 0
    offsets = []
    for (oy0, oy1) in overlay_bands:
        oc = (oy0 + oy1) / 2
        best_off = min(abs(oc - (sy0 + sy1) / 2) for (sy0, sy1) in scan_bands)
        offsets.append(best_off)
        if best_off < 15:  # within ~15px = matched to a scan line
            matched += 1

    return {
        "overlay_lines": len(overlay_bands),
        "scan_lines": len(scan_bands),
        "matched_lines": matched,
        "line_match_ratio": round(matched / len(overlay_bands), 4),
        "mean_center_offset_px": round(float(np.mean(offsets)), 2),
    }


def measure_page(overlay_rgb: np.ndarray, scan_rgb: np.ndarray) -> dict:
    """Compute alignment metrics for a single page."""
    red = _red_mask(overlay_rgb)
    dark = _dark_mask(scan_rgb)

    red_count = int(red.sum())
    dark_count = int(dark.sum())

    if red_count == 0 or dark_count == 0:
        return {
            "red_pixels": red_count,
            "dark_pixels": dark_count,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_dist_px": float("nan"),
            "median_dist_px": float("nan"),
            "line_match_ratio": 0.0,
            "mean_center_offset_px": float("nan"),
        }

    inter = int((red & dark).sum())
    union = int((red | dark).sum())
    iou = inter / union if union else 0.0
    precision = inter / red_count
    recall = inter / dark_count
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Positional error: distance from each red pixel to nearest dark pixel.
    dist = _distance_transform(dark)
    red_dists = dist[red]
    mean_dist = float(red_dists.mean())
    median_dist = float(np.median(red_dists))

    # Line-position accuracy (the metric that matters for searchable PDFs).
    line_acc = _line_position_accuracy(overlay_rgb, scan_rgb)

    return {
        "red_pixels": red_count,
        "dark_pixels": dark_count,
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_dist_px": round(mean_dist, 2),
        "median_dist_px": round(median_dist, 2),
        "line_match_ratio": line_acc["line_match_ratio"],
        "mean_center_offset_px": line_acc["mean_center_offset_px"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure overlay text alignment accuracy.")
    parser.add_argument("--overlay", required=True, type=Path, help="Overlay PDF (red debug text).")
    parser.add_argument("--scan", required=True, type=Path, help="Original scan PDF.")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default 150).")
    parser.add_argument("--pages", default="", help="Page range, e.g. '1-5' or '1,3,7'. Empty = all.")
    parser.add_argument("--offset", type=int, default=0, help="Scan page offset: overlay page i maps to scan page i+offset (0-based). Use when the overlay contains a subset of pages, e.g. --offset 30 for pages 31-35.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args()

    # Determine page range.
    overlay_doc = pymupdf.open(str(args.overlay))
    scan_doc = pymupdf.open(str(args.scan))
    n_pages = min(overlay_doc.page_count, scan_doc.page_count)
    overlay_doc.close()
    scan_doc.close()

    if args.pages:
        indices = set()
        for part in args.pages.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                indices.update(range(int(a) - 1, int(b)))
            else:
                indices.add(int(part) - 1)
        page_indices = sorted(i for i in indices if 0 <= i < n_pages)
    else:
        page_indices = list(range(n_pages))

    results = []
    for idx in page_indices:
        overlay_rgb = _render_page(args.overlay, idx, args.dpi)
        scan_rgb = _render_page(args.scan, idx + args.offset, args.dpi)
        m = measure_page(overlay_rgb, scan_rgb)
        m["page"] = idx + 1
        results.append(m)

    # Aggregate.
    if results:
        agg = {
            "pages": len(results),
            "mean_iou": round(float(np.mean([r["iou"] for r in results])), 4),
            "mean_precision": round(float(np.mean([r["precision"] for r in results])), 4),
            "mean_recall": round(float(np.mean([r["recall"] for r in results])), 4),
            "mean_f1": round(float(np.mean([r["f1"] for r in results])), 4),
            "mean_mean_dist_px": round(float(np.nanmean([r["mean_dist_px"] for r in results])), 2),
            "mean_median_dist_px": round(float(np.nanmean([r["median_dist_px"] for r in results])), 2),
            "mean_line_match_ratio": round(float(np.mean([r["line_match_ratio"] for r in results])), 4),
            "mean_center_offset_px": round(float(np.nanmean([r["mean_center_offset_px"] for r in results])), 2),
        }
    else:
        agg = {}

    if args.json:
        print(json.dumps({"per_page": results, "aggregate": agg}, indent=2))
        return

    # Human-readable table.
    print(f"{'Page':>5} {'IoU':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'LineMatch':>9} {'OffPx':>6} {'Red':>6} {'Dark':>6}")
    for r in results:
        print(
            f"{r['page']:>5} {r['iou']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['f1']:>6.3f} {r['line_match_ratio']:>9.3f} {r['mean_center_offset_px']:>6.1f} "
            f"{r['red_pixels']:>6} {r['dark_pixels']:>6}"
        )
    print("\n--- Aggregate ---")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print("\nInterpretation:")
    print("  IoU (glyph overlap): >0.7 good, 0.5-0.7 fair, <0.5 poor")
    print("  LineMatch (line-position): >0.8 good — this is what matters for searchable PDFs")


if __name__ == "__main__":
    main()
