from __future__ import annotations

import numpy as np
import cv2


def create_depth_threshold_mask(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    if depth_m.dtype != np.float32 and depth_m.dtype != np.float64:
        depth = depth_m.astype(np.float32)
    else:
        depth = depth_m
    valid = np.isfinite(depth)
    mask = (depth >= float(min_m)) & (depth <= float(max_m)) & valid
    return mask.astype(np.uint8) * 255


def apply_circular_roi(mask: np.ndarray, cx: int | None, cy: int | None, r: int | None) -> np.ndarray:
    if cx is None or cy is None or r is None or r <= 0:
        return mask
    h, w = mask.shape[:2]
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi, (int(cx), int(cy)), int(r), 255, thickness=-1)
    return cv2.bitwise_and(mask, roi)


def morphology_open_close(mask: np.ndarray, k_open: int = 0, k_close: int = 0) -> np.ndarray:
    result = mask
    if k_open and k_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if k_close and k_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    # Flood fill from border to find background, then invert
    h, w = mask.shape[:2]
    filled = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood_mask, seedPoint=(0, 0), newVal=255)
    inv = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, inv)


def apply_mask_to_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Return RGBA where alpha is mask
    if rgb.dtype != np.uint8:
        img = rgb.astype(np.uint8)
    else:
        img = rgb
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if mask.ndim == 3:
        alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        alpha = mask
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return np.dstack([img, alpha])


