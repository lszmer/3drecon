import argparse
from pathlib import Path
import shutil

from dataio.data_io import DataIO
from models.side import Side


def write_run_script(root: Path, camera_model: str, camera_params: str):
    db = root / "database.db"
    imgs = root / "images"
    sparse = root / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    script = root / "colmap_run.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"colmap feature_extractor --database_path {db} --image_path {imgs} --ImageReader.camera_model {camera_model} --ImageReader.camera_params {camera_params}",
        f"colmap sequential_matcher --database_path {db} --SequentialMatching.overlap 5",
        f"colmap mapper --database_path {db} --image_path {imgs} --output_path {sparse}",
        "echo 'Done. Open the reconstruction in COLMAP GUI from the sparse folder.'",
        ""
    ]
    script.write_text("\n".join(lines), encoding="utf-8")
    try:
        script.chmod(0o755)
    except Exception:
        pass


def export_side(data_io: DataIO, side: Side, out_root: Path):
    out_images = out_root / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # Read intrinsics from camera characteristics
    ch = data_io.color.load_camera_characteristics(side=side)
    camera_model = "PINHOLE"  # Use pinhole; adjust if you want SIMPLE_RADIAL
    camera_params = f"{ch.fx},{ch.fy},{ch.cx},{ch.cy}"

    # Copy RGB PNGs
    rgb_paths = data_io.color.image_path_config.get_rgb_image_paths(side=side)
    for p in rgb_paths:
        ts = p.stem
        dst = out_images / f"{side.name}_{ts}.png"
        try:
            shutil.copy2(p, dst)
        except Exception:
            continue

    write_run_script(out_root, camera_model=camera_model, camera_params=camera_params)


def main():
    ap = argparse.ArgumentParser(description="Export turntable object images and intrinsics for COLMAP")
    ap.add_argument("--project_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    out_left = args.output_dir / "LEFT"
    out_right = args.output_dir / "RIGHT"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    data_io = DataIO(project_dir=args.project_dir)
    export_side(data_io, Side.LEFT, out_left)
    export_side(data_io, Side.RIGHT, out_right)

    print(f"[Info] Exported COLMAP-ready folders:\n  LEFT: {out_left}\n  RIGHT: {out_right}")
    print("[Info] Next: run LEFT/colmap_run.sh (and RIGHT/colmap_run.sh if desired).")


if __name__ == "__main__":
    main()


