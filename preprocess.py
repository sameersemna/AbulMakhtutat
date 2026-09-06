"""OpenCV-based image preprocessing for scanned Arabic pages.

Applied *before* the vision OCR call to improve accuracy on degraded scans:
skew correction, adaptive binarization, denoising, and contrast normalization.

All steps are optional and individually toggleable via config, so the pipeline
can A/B test whether preprocessing helps the vision model (which is already
fairly robust). The module degrades gracefully: if OpenCV is unavailable, the
original image is returned unchanged.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from config import Config, get_config

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False


def _pil_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to a BGR numpy array via PIL."""
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(im)[:, :, ::-1].copy()  # RGB -> BGR


def _bgr_to_png(bgr: np.ndarray) -> bytes:
    """Encode a BGR numpy array back to PNG bytes."""
    rgb = bgr[:, :, ::-1]
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _deskew(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct small rotation using the min-area rectangle of text pixels.

    Returns (rotated_image, angle_degrees). Skips if the detected angle is
    below a threshold to avoid over-rotating near-straight pages.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Invert so text becomes white foreground on black background.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 100:
        return bgr, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # Normalize angle to the range [-45, 45].
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.3:
        return bgr, 0.0

    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def _adaptive_binarize(bgr: np.ndarray) -> np.ndarray:
    """Adaptive thresholding to clean uneven illumination / bleed-through."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Mild blur to reduce noise before thresholding.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _denoise(bgr: np.ndarray) -> np.ndarray:
    """Non-local means denoising (preserves text edges better than blur)."""
    return cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)


def _normalize_contrast(bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast normalization on the L channel (LAB colorspace)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _light_contrast(bgr: np.ndarray) -> np.ndarray:
    """Cheap global contrast normalization (no deskew/binarize/denoise).

    Stretches the grayscale histogram to the full range and applies a mild
    CLAHE pass. Fast and low-risk — intended for the "light" preprocessing mode.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Global histogram stretch to use the full brightness range.
    min_val, max_val = gray.min(), gray.max()
    if max_val > min_val:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # Mild local contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess(image_bytes: bytes, config: Config | None = None) -> bytes:
    """Apply the configured preprocessing pipeline to a page image.

    Returns the (possibly transformed) PNG bytes. If OpenCV is missing or the
    mode is "off", returns the original bytes unchanged.
    """
    config = config or get_config()
    pp = config.pipeline.preprocess

    if not _HAS_CV2:
        logger.warning("OpenCV not available; skipping preprocessing")
        return image_bytes

    mode = pp.mode.lower()
    if mode == "off" and not pp.enabled:
        return image_bytes

    try:
        bgr = _pil_to_bgr(image_bytes)

        if mode == "light":
            # Cheap global contrast normalization only.
            bgr = _light_contrast(bgr)
            return _bgr_to_png(bgr)

        # "full" mode (or legacy per-flag enablement).
        if pp.deskew:
            bgr, angle = _deskew(bgr)
            if angle:
                logger.debug("Deskewed page by %.2f degrees", angle)
        if pp.binarize:
            bgr = _adaptive_binarize(bgr)
        if pp.denoise:
            bgr = _denoise(bgr)
        if pp.normalize_contrast:
            bgr = _normalize_contrast(bgr)
        return _bgr_to_png(bgr)
    except Exception as exc:
        logger.warning("Preprocessing failed, using original image: %s", exc)
        return image_bytes


def enhance_contrast(image_bytes: bytes) -> bytes:
    """Enhance contrast for washed-out / overexposed scans.

    Used as a retry strategy when the vision model reports the page is
    "overexposed" or "washed out". Converts to grayscale and applies CLAHE
    (local contrast enhancement) plus a global histogram stretch to recover
    faint text. Returns the original bytes unchanged if OpenCV is unavailable
    or the enhancement fails.
    """
    if not _HAS_CV2:
        return image_bytes
    try:
        bgr = _pil_to_bgr(image_bytes)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Global auto-contrast: stretch the histogram to use the full range.
        # This darkens washed-out (too-bright) pages and brightens dark ones.
        min_val, max_val = gray.min(), gray.max()
        if max_val > min_val:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Local contrast enhancement (CLAHE) to recover faint text.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return _bgr_to_png(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
    except Exception as exc:
        logger.warning("Contrast enhancement failed, using original image: %s", exc)
        return image_bytes


def detect_word_boxes(image_bytes: bytes) -> list[tuple[int, int, int, int]]:
    """Detect word bounding boxes in a page image via connected components.

    Arabic is cursive, so each word is a single connected component. We binarize
    the image, find connected components, and return their bounding boxes in
    reading order (top-to-bottom, then right-to-left within each line).

    Returns a list of (x0, y0, x1, y1) boxes in pixel coordinates. Returns an
    empty list if OpenCV is unavailable or detection fails.
    """
    if not _HAS_CV2:
        return []
    try:
        bgr = _pil_to_bgr(image_bytes)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Binarize: text becomes white foreground on black background.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate slightly to merge diacritics with their base word.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        # Find connected components (words).
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        boxes: list[tuple[int, int, int, int]] = []
        h, w = binary.shape
        min_area = 30  # ignore tiny specks
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < min_area:
                continue
            # Ignore components that are too large (likely borders/artifacts).
            if bw > w * 0.9 and bh > h * 0.9:
                continue
            boxes.append((x, y, x + bw, y + bh))

        # Sort in reading order: top-to-bottom by line, then right-to-left.
        # Group by vertical center into lines.
        boxes.sort(key=lambda b: (b[1], -b[0]))  # y asc, x desc (RTL)
        return boxes
    except Exception as exc:
        logger.warning("Word box detection failed: %s", exc)
        return []


def detect_line_boxes(image_bytes: bytes) -> list[tuple[int, int, int, int]]:
    """Detect text-line bounding boxes by grouping word boxes into lines.

    Returns a list of (x0, y0, x1, y1) line boxes in top-to-bottom reading order.
    More robust than word-level for mapping OCR text (line breaks roughly
    correspond to visual lines). Returns [] if OpenCV is unavailable.
    """
    word_boxes = detect_word_boxes(image_bytes)
    if not word_boxes:
        return []

    # Group word boxes into lines by vertical overlap of their y-ranges.
    # Two boxes belong to the same line if their vertical ranges overlap by a
    # meaningful amount (>= 50% of the smaller box's height).
    boxes = sorted(word_boxes, key=lambda b: b[1])  # sort by y0
    lines: list[list[tuple[int, int, int, int]]] = []

    for box in boxes:
        x0, y0, x1, y1 = box
        bh = y1 - y0
        placed = False
        for line in lines:
            # Compute the line's current vertical extent.
            ly0 = min(b[1] for b in line)
            ly1 = max(b[3] for b in line)
            # Vertical overlap between this box and the line.
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap > 0 and overlap >= 0.5 * min(bh, ly1 - ly0):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    # Merge each line's boxes into a single bounding box.
    line_boxes = []
    for line in lines:
        x0 = min(b[0] for b in line)
        y0 = min(b[1] for b in line)
        x1 = max(b[2] for b in line)
        y1 = max(b[3] for b in line)
        line_boxes.append((int(x0), int(y0), int(x1), int(y1)))

    # Filter out tiny boxes (diacritics/dots detected as separate lines).
    # A real text line is at least ~30px tall at 300 DPI.
    line_boxes = [b for b in line_boxes if (b[3] - b[1]) >= 30]

    # Sort top-to-bottom.
    line_boxes.sort(key=lambda b: b[1])
    return line_boxes


def detect_words_reading_order(image_bytes: bytes) -> list[tuple[int, int, int, int]]:
    """Detect real word boxes in reading order (top-to-bottom, RTL within line).

    Unlike `detect_word_boxes` (which returns raw connected components including
    diacritics/dots), this filters to actual words and returns them in the order
    a reader would read them. Used for pixel-accurate overlay alignment.

    Returns a list of (x0, y0, x1, y1) boxes in reading order.
    """
    if not _HAS_CV2:
        return []
    try:
        bgr = _pil_to_bgr(image_bytes)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate to merge diacritics with their base word.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        h, w = binary.shape
        # Filter to real words: drop tiny diacritics/dots and full-page artifacts.
        min_area = 200  # px² at 300 DPI
        min_height = 20  # px
        words = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < min_area or bh < min_height:
                continue
            if bw > w * 0.9 and bh > h * 0.9:
                continue
            words.append((int(x), int(y), int(x + bw), int(y + bh)))

        # Group into lines by vertical overlap.
        words.sort(key=lambda b: b[1])
        lines: list[list[tuple[int, int, int, int]]] = []
        for box in words:
            x0, y0, x1, y1 = box
            bh = y1 - y0
            placed = False
            for line in lines:
                ly0 = min(b[1] for b in line)
                ly1 = max(b[3] for b in line)
                overlap = min(y1, ly1) - max(y0, ly0)
                if overlap > 0 and overlap >= 0.5 * min(bh, ly1 - ly0):
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        # Sort lines top-to-bottom, and words within each line right-to-left (RTL).
        lines.sort(key=lambda l: min(b[1] for b in l))
        result = []
        for line in lines:
            line.sort(key=lambda b: -b[0])  # descending x = right-to-left
            result.extend(line)
        return result
    except Exception as exc:
        logger.warning("Word reading-order detection failed: %s", exc)
        return []
