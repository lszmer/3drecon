import argparse
from pathlib import Path
import subprocess
import re

def find_latest_session(base_dir: Path):
    candidates = [d for d in base_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
    if not candidates:
        raise RuntimeError(f"No session directory matching pattern found in {base_dir}")
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return latest

def run_pipeline(project_dir: Path):
    base_cmd = [
        ("scripts/convert_yuv_to_rgb.py", []),
        ("scripts/convert_depth_to_linear_map.py", []),
        ("scripts/reconstruct_scene.py",
         ["--config", "config/pipeline_config.yml"]),
        ("scripts/build_colmap_project.py",
         ["--output_dir", str(project_dir / "COLMAP"),
          "--use_colored_pointcloud",
          "--use_optimized_color_dataset",
          "--interval", "1"]),
        ("scripts/visualize_confidence_maps.py", [])
    ]

    for script, extra_args in base_cmd:
        cmd = ["python", script, "--project_dir", str(project_dir)]
        if "config" not in cmd and any(["config" in a for a in extra_args]):
            cmd += ["--config", "config/pipeline_config.yml"]
        cmd += extra_args
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, help="Path to project/session directory")
    parser.add_argument("--session_dir", type=Path, help="Specify session dir directly. If not given, auto-select latest.")
    args = parser.parse_args()

    if args.session_dir:
        project_dir = args.session_dir
    elif args.project_dir:
        # look for latest session in given project dir
        project_dir = find_latest_session(args.project_dir)
        print(f"[Info] No --session_dir specified. Found latest session: {project_dir}")
    else:
        # Use hardcoded base folder
        qrc_root = Path("/Users/linus/Documents/QuestRealityCapture")
        project_dir = find_latest_session(qrc_root)
        print(f"[Info] No --project_dir or --session_dir specified. Auto-picked {project_dir}")

    run_pipeline(project_dir)

if __name__ == "__main__":
    main()
