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


def preprocess(image_bytes: bytes, config: Config | None = None) -> bytes:
    """Apply the configured preprocessing pipeline to a page image.

    Returns the (possibly transformed) PNG bytes. If OpenCV is missing or all
    steps are disabled, returns the original bytes unchanged.
    """
    config = config or get_config()
    pp = config.pipeline.preprocess

    if not _HAS_CV2:
        logger.warning("OpenCV not available; skipping preprocessing")
        return image_bytes
    if not pp.enabled:
        return image_bytes

    try:
        bgr = _pil_to_bgr(image_bytes)
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
