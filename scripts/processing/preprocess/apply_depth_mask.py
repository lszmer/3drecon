import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import yaml
import numpy as np

from dataio.data_io import DataIO
from models.side import Side

from processing.utils.mask_utils import (
    create_depth_threshold_mask,
    apply_circular_roi,
    morphology_open_close,
    fill_holes,
    apply_mask_to_rgb,
)


def load_masking_config(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    masking = (cfg.get("masking") or {})
    preprocessing = (cfg.get("preprocessing") or {})

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

    return min_m, max_m, k_open, k_close, fill, cx, cy, r


def ensure_dirs(out_mask_dir: Path, out_rgba_dir: Path):
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_rgba_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Apply depth-based masking using color-aligned depth to mask RGB.")
    parser.add_argument("--project_dir", type=Path, required=True, help="Session or subset directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sides", type=str, default="left,right", help="Comma-separated sides: left,right")
    args = parser.parse_args()

    min_m, max_m, k_open, k_close, do_fill, cx, cy, r = load_masking_config(args.config)
    if min_m is None or max_m is None:
        raise ValueError("masking.depth_threshold_m.min/max must be set in config for masking")

    sides = [s.strip() for s in args.sides.split(',') if s.strip()]
    for side in sides:
        rgb_dir = args.project_dir / ("left_camera_rgb" if side == "left" else "right_camera_rgb")
        depth_lin_dir = args.project_dir / ("left_depth_linear" if side == "left" else "right_depth_linear")
        depth_color_aligned_dir = args.project_dir / ("left_color_aligned_depth" if side == "left" else "right_color_aligned_depth")

        # Output directories
        out_depth_mask_dir = args.project_dir / ("left_depth_masks" if side == "left" else "right_depth_masks")
        out_rgba_dir = args.project_dir / ("left_camera_rgba" if side == "left" else "right_camera_rgba")

        # Ensure output dirs (RGBA only if we actually write RGBA later)
        out_depth_mask_dir.mkdir(parents=True, exist_ok=True)

        # Optional: read camera format to infer image size for ROI sanity
        fmt_json = args.project_dir / ("left_camera_image_format.json" if side == "left" else "right_camera_image_format.json")
        if fmt_json.exists():
            try:
                info = json.loads(fmt_json.read_text())
                _ = info.get("width"), info.get("height")
            except Exception:
                pass

        data_io = DataIO(project_dir=args.project_dir)
        # Prepare datasets for on-the-fly alignment if needed
        depth_ds = data_io.depth.load_depth_dataset(side=Side.LEFT if side == "left" else Side.RIGHT, use_cache=True)
        color_ds = data_io.color.load_color_dataset(side=Side.LEFT if side == "left" else Side.RIGHT, use_cache=True)
        Kc_all = color_ds.get_intrinsic_matrices()
        Td_cw_all = depth_ds.transforms.extrinsics_cw  # camera-to-world
        Tc_wc_all = color_ds.transforms.extrinsics_wc  # world-to-camera

        # Ensure output dir for aligned depth exists (for caching)
        depth_color_aligned_dir.mkdir(parents=True, exist_ok=True)

        out_rgba_dir.mkdir(parents=True, exist_ok=True)
        rgb_files = sorted(rgb_dir.glob("*.png"))
        for rgb_path in rgb_files:
            ts = rgb_path.stem
            depth_path = depth_color_aligned_dir / f"{ts}.npy"
            if not depth_path.exists():
                # Compute color-aligned depth on-the-fly using dataset transforms
                try:
                    tsi = int(ts)
                except Exception:
                    continue
                try:
                    di = depth_ds.find_nearest_index(tsi)
                    ci = color_ds.find_nearest_index(tsi)
                except Exception:
                    continue
                depth_map = data_io.depth.load_depth_map_by_index(side=Side.LEFT if side == "left" else Side.RIGHT, dataset=depth_ds, index=di)
                if depth_map is None:
                    continue
                fx_d = float(depth_ds.fx[di]); fy_d = float(depth_ds.fy[di])
                cx_d = float(depth_ds.cx[di]); cy_d = float(depth_ds.cy[di])
                Kc = Kc_all[ci]
                # Backproject depth to depth cam points
                h, w = depth_map.shape
                us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                z = depth_map.astype(np.float32)
                Xd = (us - cx_d) * z / fx_d
                Yd = (vs - cy_d) * z / fy_d
                pts_d = np.stack([Xd, Yd, z, np.ones_like(z)], axis=-1)
                # Depth cam -> world -> color cam
                Td_cw = Td_cw_all[di]
                Tc_wc = Tc_wc_all[ci]
                pts_d_flat = pts_d.reshape(-1, 4).T
                pts_w = (Td_cw @ pts_d_flat)
                pts_c = (Tc_wc @ pts_w).T.reshape(h, w, 4)[..., :3]
                # Rasterize to color image size
                color_h = int(color_ds.heights[ci]); color_w = int(color_ds.widths[ci])
                out = np.full((color_h, color_w), 0.0, dtype=np.float32)
                Xc = pts_c[..., 0]; Yc = pts_c[..., 1]; Zc = pts_c[..., 2]
                valid = Zc > 0
                u = (Kc[0, 0] * (Xc / Zc) + Kc[0, 2]).round().astype(np.int32)
                v = (Kc[1, 1] * (Yc / Zc) + Kc[1, 2]).round().astype(np.int32)
                inb = valid & (u >= 0) & (u < color_w) & (v >= 0) & (v < color_h)
                u = u[inb]; v = v[inb]; zc = Zc[inb]
                # Simple z-buffer
                zbuf = np.full((color_h, color_w), np.inf, dtype=np.float32)
                for ui, vi, zi in zip(u, v, zc):
                    if zi < zbuf[vi, ui]:
                        zbuf[vi, ui] = zi
                        out[vi, ui] = zi
                np.save(depth_path, out)
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            try:
                depth_m = np.load(depth_path)
            except Exception:
                continue
            if depth_m is None or rgb_bgr is None:
                continue
            if depth_m.dtype != np.float32:
                depth_m = depth_m.astype(np.float32)

            mask = create_depth_threshold_mask(depth_m, min_m=min_m, max_m=max_m)
            mask = apply_circular_roi(mask, cx=cx, cy=cy, r=r)
            mask = morphology_open_close(mask, k_open=k_open, k_close=k_close)
            if do_fill:
                mask = fill_holes(mask)

            rgba = apply_mask_to_rgb(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), mask)
            rgba_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

            cv2.imwrite(str(out_depth_mask_dir / f"{ts}.png"), mask)
            cv2.imwrite(str(out_rgba_dir / f"{ts}.png"), rgba_bgra)

        print(f"[Info] Wrote depth-frame masks: {out_depth_mask_dir}")
        print(f"[Info] Wrote masked RGBA: {out_rgba_dir}")


if __name__ == "__main__":
    main()


