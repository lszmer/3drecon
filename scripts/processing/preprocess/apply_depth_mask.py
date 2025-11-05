import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import yaml
import sys
from pathlib import Path as _P

# Ensure 'scripts' package root on sys.path (to mirror other scene scripts)
_SCRIPTS_ROOT = _P(__file__).resolve().parents[2]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from processing.utils.mask_utils import (
    create_depth_threshold_mask,
    apply_circular_roi,
    morphology_open_close,
    fill_holes,
)


def load_masking_config(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    masking = (cfg.get("masking") or {})
    preprocessing = (cfg.get("preprocessing") or {})
    depth_to_linear = (cfg.get("depth_to_linear") or {})

    depth_thr = masking.get("depth_threshold_m", {})
    min_m = depth_thr.get("min", None)
    max_m = depth_thr.get("max", None)

    morph = masking.get("morphology", {})
    k_open = int(morph.get("open", 0) or 0)
    k_close = int(morph.get("close", 0) or 0)
    fill = bool(masking.get("fill_holes", True))

    circ = preprocessing.get("circular_roi_pixels", {})
    cx = circ.get("cx", None)
    cy = circ.get("cy", None)
    r = circ.get("r", None)

    clip_near_m = float(depth_to_linear.get("clip_near_m", 0.1) or 0.1)
    clip_far_m = float(depth_to_linear.get("clip_far_m", 3.0) or 3.0)

    return min_m, max_m, k_open, k_close, fill, cx, cy, r, clip_near_m, clip_far_m


def ensure_dirs(out_mask_dir: Path, out_rgba_dir: Path):
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_rgba_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Apply depth-based masking directly on linear depth. No RGB processing.")
    parser.add_argument("--project_dir", type=Path, required=True, help="Session or subset directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sides", type=str, default="left,right", help="Comma-separated sides: left,right")
    args = parser.parse_args()

    min_m, max_m, k_open, k_close, do_fill, cx, cy, r, clip_near_m, clip_far_m = load_masking_config(args.config)
    if min_m is None or max_m is None:
        raise ValueError("masking.depth_threshold_m.min/max must be set in config for masking")

    sides = [s.strip() for s in args.sides.split(',') if s.strip()]
    from dataio.data_io import DataIO
    from models.side import Side
    data_io = DataIO(project_dir=args.project_dir)

    for side in sides:
        side_enum = Side.LEFT if side == "left" else Side.RIGHT
        out_depth_masked_dir = args.project_dir / ("left_depth_linear_masked" if side == "left" else "right_depth_linear_masked")

        out_depth_masked_dir.mkdir(parents=True, exist_ok=True)

        depth_ds = data_io.depth.load_depth_dataset(side=side_enum, use_cache=True)
        total = len(depth_ds.timestamps)
        written = 0
        failed = 0
        empty_masks = 0

        for i in range(total):
            ts = int(depth_ds.timestamps[i])
            depth_m = data_io.depth.load_depth_map_by_index(side=side_enum, dataset=depth_ds, index=i)
            if depth_m is None:
                failed += 1
                continue

            # Build strict threshold mask first (uint8 0/255)
            thresh_mask = create_depth_threshold_mask(depth_m, min_m=min_m, max_m=max_m)
            # Apply optional ROI and morphology on a working mask
            morph_mask = apply_circular_roi(thresh_mask, cx=cx, cy=cy, r=r)
            morph_mask = morphology_open_close(morph_mask, k_open=k_open, k_close=k_close)
            if do_fill:
                morph_mask = fill_holes(morph_mask)
            # Enforce threshold again after morphology to avoid leaking outside range
            mask = cv2.bitwise_and(morph_mask, thresh_mask)

            nonzero = int(np.count_nonzero(mask))
            if nonzero == 0:
                empty_masks += 1

            masked_depth = depth_m.copy()
            masked_depth[mask == 0] = 0.0
            # Clamp depths to the configured mask range for consistency
            masked_depth = np.clip(masked_depth, min_m, max_m)

            # Save metric masked depth as .npy (float32 meters)
            np.save(str(out_depth_masked_dir / f"{ts}.npy"), masked_depth.astype(np.float32))

            lin01 = np.clip((masked_depth - clip_near_m) / max(1e-6, (clip_far_m - clip_near_m)), 0.0, 1.0)
            viz_u8 = (lin01 * 255.0).astype(np.uint8)
            cv2.imwrite(str(out_depth_masked_dir / f"{ts}_viz.png"), viz_u8)

            written += 1

        print(f"[Info] Side={side}: total={total}, written={written}, empty_masks={empty_masks}, failed_loads={failed}")
        print(f"[Info] Wrote masked depth (.npy meters) and viz (u8) to: {out_depth_masked_dir}")


if __name__ == "__main__":
    main()


