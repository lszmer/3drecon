import argparse
import shutil
from pathlib import Path
from typing import List, Set

import yaml


def load_frame_selection(config_path: Path):
    try:
        data = yaml.safe_load(config_path.read_text())
        fs = data.get("frame_selection", {}) or {}
        return int(fs.get("sample_by_time_step", 0) or 0), int(fs.get("target_frame_count", 0) or 0)
    except Exception:
        return 0, 0


def pick_indices(total: int, sample_step: int, target_count: int) -> Set[int]:
    if total == 0:
        return set()
    if sample_step and sample_step > 1:
        return set(range(0, total, sample_step))
    if target_count and target_count < total:
        # even sampling
        step = max(1, total // target_count)
        return set(range(0, total, step))
    return set(range(total))


def timestamps_from_files(files: List[Path]) -> List[int]:
    ts = []
    for p in files:
        try:
            ts.append(int(p.stem))
        except Exception:
            continue
    ts.sort()
    return ts


def copy_selected(files: List[Path], selected_set: Set[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        try:
            ts = int(p.stem)
        except Exception:
            continue
        if ts in selected_set:
            shutil.copy2(p, out_dir / p.name)


def filter_depth_descriptors(csv_path: Path, selected_set: Set[int], out_csv: Path):
    if not csv_path.exists():
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Expect column name 'timestamp_ms'
    if 'timestamp_ms' not in df.columns:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return
    df_f = df[df['timestamp_ms'].astype(int).isin(selected_set)].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Turntable preprocess: downsample frames and prepare subset session")
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sample_step", type=int, default=None)
    parser.add_argument("--target_count", type=int, default=None)
    parser.add_argument("--preset", type=str, default=None)
    args = parser.parse_args()

    # Load defaults from config; allow CLI overrides
    cfg_sample_step, cfg_target_count = load_frame_selection(args.config)
    sample_step = args.sample_step if args.sample_step is not None else cfg_sample_step
    target_count = args.target_count if args.target_count is not None else cfg_target_count

    # Source dirs
    L_RGB = args.project_dir / "left_camera_rgb"
    R_RGB = args.project_dir / "right_camera_rgb"
    L_DEPTH = args.project_dir / "left_depth"
    R_DEPTH = args.project_dir / "right_depth"

    # Create subset root
    subset = args.out_dir
    subset.mkdir(parents=True, exist_ok=True)

    # Copy camera metadata if present
    for meta in [
        "left_camera_image_format.json",
        "right_camera_image_format.json",
        "left_camera_characteristics.json",
        "right_camera_characteristics.json",
        "hmd_poses.csv",
    ]:
        src = args.project_dir / meta
        if src.exists():
            shutil.copy2(src, subset / meta)

    # RGB downsample per side (if present)
    selected_ts: Set[int] = set()
    for side_dir, out_name in [
        (L_RGB, "left_camera_rgb"),
        (R_RGB, "right_camera_rgb"),
    ]:
        if not side_dir.exists():
            continue
        files = sorted(side_dir.glob("*.png"))
        ts_list = timestamps_from_files(files)
        idxs = pick_indices(len(ts_list), sample_step, target_count)
        selected_ts = set(ts_list[i] for i in idxs)
        copy_selected(files, selected_ts, subset / out_name)

    # Depth downsample matching selected timestamps (if any RGB selected)
    for depth_dir, out_name, desc_csv in [
        (L_DEPTH, "left_depth", "left_depth_descriptors.csv"),
        (R_DEPTH, "right_depth", "right_depth_descriptors.csv"),
    ]:
        if not depth_dir.exists():
            continue
        files = sorted(depth_dir.glob("*.raw"))
        if selected_ts:
            copy_selected(files, selected_ts, subset / out_name)
            filter_depth_descriptors(args.project_dir / desc_csv, selected_ts, subset / desc_csv)
        else:
            # If no RGB, fall back to even sampling on depth
            ts_list = timestamps_from_files(files)
            idxs = pick_indices(len(ts_list), sample_step, target_count)
            depth_selected = set(ts_list[i] for i in idxs)
            copy_selected(files, depth_selected, subset / out_name)
            filter_depth_descriptors(args.project_dir / desc_csv, depth_selected, subset / desc_csv)

    # Record subset metadata
    meta = {
        "source": str(args.project_dir),
        "sample_step": sample_step,
        "target_count": target_count,
        "preset": args.preset,
    }
    (subset / "subset_meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    print(f"[Info] Subset prepared at: {subset}")


if __name__ == "__main__":
    main()


