import argparse
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Turntable Object Reconstruction Runner")
    parser.add_argument("--project_dir", type=Path, required=True, help="Path to the source session directory")
    parser.add_argument("--config", type=Path, default=Path("config/object_turntable.yml"), help="Config YAML path")
    parser.add_argument("--stages", type=str, default="preprocess,convert,depth,mask,reconstruct", help="Comma-separated stages: preprocess,convert,depth,mask,reconstruct,colmap")
    parser.add_argument("--preset", type=str, default=None, help="Optional preset name to record in subset metadata")
    parser.add_argument("--sample_step", type=int, default=None, help="Override: keep 1 of every N frames (preprocess)")
    parser.add_argument("--target_count", type=int, default=None, help="Override: target frame count (preprocess)")
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        print(f"[Error] Project directory does not exist: {args.project_dir}")
        sys.exit(2)

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]

    # Preprocess builds a subset directory; subsequent stages consume it
    subset_dir = args.project_dir / "subset"

    if "preprocess" in stages:
        cmd = [
            sys.executable, "scripts/preprocess_turntable.py",
            "--project_dir", str(args.project_dir),
            "--config", str(args.config),
            "--out_dir", str(subset_dir),
        ]
        if args.sample_step is not None:
            cmd += ["--sample_step", str(args.sample_step)]
        if args.target_count is not None:
            cmd += ["--target_count", str(args.target_count)]
        if args.preset:
            cmd += ["--preset", args.preset]
        run(cmd)

    # Use subset if it exists; otherwise fall back to source dir
    work_dir = subset_dir if subset_dir.is_dir() else args.project_dir

    if "convert" in stages:
        run([sys.executable, "scripts/convert_yuv_to_rgb.py", "--project_dir", str(work_dir), "--config", str(args.config)])

    if "depth" in stages:
        run([sys.executable, "scripts/convert_depth_to_linear_map.py", "--project_dir", str(work_dir), "--config", str(args.config)])

    if "mask" in stages:
        run([sys.executable, "scripts/processing/preprocess/apply_depth_mask.py", "--project_dir", str(work_dir), "--config", str(args.config)])

    if "reconstruct" in stages:
        run([sys.executable, "scripts/reconstruct_scene.py", "--project_dir", str(work_dir), "--config", str(args.config)])

    if "colmap" in stages:
        run([sys.executable, "scripts/build_colmap_project.py", "--project_dir", str(work_dir), "--output_dir", str(work_dir / "COLMAP"), "--use_colored_pointcloud", "--use_optimized_color_dataset", "--interval", "1"])


if __name__ == "__main__":
    main()


