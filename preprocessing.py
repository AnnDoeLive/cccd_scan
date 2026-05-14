"""
Image Preprocessing Pipeline for CCCD OCR
==========================================

Pipeline:  BGR → Grayscale → Deskew (Hough) → Morphology → OCR-ready binary

Theory summary
--------------
Stage 1 – Deskew via Hough Transform
  • Canny edge detection finds pixel-level edges (gradient magnitude thresholding).
  • Probabilistic Hough Line Transform (HoughLinesP) votes in (rho, theta) space;
    each edge pixel "votes" for every line it could belong to.  Lines that accumulate
    enough votes become candidates.
  • Near-horizontal lines (|angle| < 45°) represent text baselines.
  • The median angle of all candidate lines is the document skew angle.
  • A rotation matrix R(θ) is applied to warpAffine the image back to 0°.

Stage 2 – Morphology + OCR Optimization
  • CLAHE (Contrast Limited Adaptive Histogram Equalization): splits the image into
    tiles and equalises each histogram locally, capped at clipLimit to prevent noise
    amplification.  Result: uniform contrast despite uneven lighting.
  • Gaussian Blur (σ auto from 3×3 kernel): convolves with Gaussian kernel to
    attenuate high-frequency noise before thresholding.
  • Otsu Binarization: maximises inter-class variance between foreground (text) and
    background pixels; fully automatic threshold selection.
  • Morphological Opening (erosion ∘ dilation, 3×3 rect):
      – Erosion shrinks bright blobs → removes isolated noise pixels smaller than kernel.
      – Dilation restores the eroded character body.
  • Morphological Closing (dilation ∘ erosion, 2×2 rect):
      – Dilation bridges small breaks inside character strokes.
      – Erosion shrinks back to original boundary, leaving filled characters.
  • Upscale (if height < 64 px): EasyOCR's CNN feature extractor is trained on text
    lines ≥ 32 px tall; bilinear upscaling preserves smooth stroke edges.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Deskew via Hough Transform
# ---------------------------------------------------------------------------

def deskew_hough(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect and correct image skew using the Probabilistic Hough Transform.

    Parameters
    ----------
    gray : H×W uint8 grayscale image

    Returns
    -------
    deskewed : H×W uint8 straightened grayscale image
    angle    : detected skew angle in degrees (positive = counter-clockwise tilt)
    """
    # --- Edge detection (prerequisite for Hough) ---
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # --- Probabilistic Hough Line Transform ---
    # rho=1 px resolution, theta=1° resolution
    # threshold=100 votes needed, minLineLength=100 px, maxLineGap=10 px
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )

    if lines is None:
        return gray, 0.0

    # --- Collect angles of near-horizontal lines only ---
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45.0 <= angle <= 45.0:          # discard near-vertical lines
            angles.append(angle)

    if not angles:
        return gray, 0.0

    skew_angle = float(np.median(angles))

    if abs(skew_angle) < 0.3:               # negligible skew – skip rotation
        return gray, skew_angle

    # --- Rotate to correct skew ---
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=skew_angle, scale=1.0)
    deskewed = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,    # fill border with edge pixels
    )
    return deskewed, skew_angle


# ---------------------------------------------------------------------------
# Stage 2: Morphology + OCR Optimisation
# ---------------------------------------------------------------------------

def morphology_ocr_optimize(gray: np.ndarray) -> np.ndarray:
    """
    Apply contrast enhancement and denoising for VietOCR.
    Otsu binarisation and Morphological Opening are intentionally omitted:
    - Otsu hard-binarizes the image, destroying Vietnamese diacritic marks.
    - Opening (3×3 erosion) erodes thin strokes of diacritics (ắ→a, ồ→o).
    VietOCR's CNN expects natural grayscale, not a binary image.

    Parameters
    ----------
    gray : H×W uint8 grayscale image (already deskewed)

    Returns
    -------
    result : H'×W' uint8 grayscale image, ready for VietOCR
    """
    # 1. CLAHE – local contrast equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gaussian blur – suppress high-frequency noise
    blurred = cv2.GaussianBlur(enhanced, ksize=(3, 3), sigmaX=0)

    # 3. Morphological Closing (2×2) – fill micro-gaps inside strokes only
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 4. Upscale if region is too small for the OCR CNN
    h, w = closed.shape
    if h < 64:
        scale = 64.0 / h
        new_w = max(1, int(w * scale))
        closed = cv2.resize(closed, (new_w, 64), interpolation=cv2.INTER_LINEAR)

    return closed


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_for_ocr(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    End-to-end preprocessing: BGR crop → OCR-ready binary image.

    Steps
    -----
    1. BGR → Grayscale
    2. Deskew with Hough Transform
    3. Morphology + OCR optimisation

    Parameters
    ----------
    img_bgr : H×W×3 uint8 BGR image (numpy array from OpenCV or PIL→numpy)

    Returns
    -------
    processed : H'×W' uint8 binary image
    skew_angle : float, detected skew angle in degrees
    """
    # Step 1 – convert colour space
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2 – deskew
    deskewed, skew_angle = deskew_hough(gray)

    # Step 3 – morphology / OCR optimisation
    processed = morphology_ocr_optimize(deskewed)

    return processed, skew_angle
