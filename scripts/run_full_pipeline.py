import argparse
from pathlib import Path
import subprocess
import re
import time

def find_latest_session(base_dir: Path):
    candidates = [d for d in base_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
    if not candidates:
        raise RuntimeError(f"No session directory matching pattern found in {base_dir}")
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return latest

def run_pipeline(project_dir: Path) -> float:
    view_seconds = 0.0
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
        #("scripts/visualize_confidence_maps.py", [])
    ]

    for script, extra_args in base_cmd:
        cmd = ["python", script, "--project_dir", str(project_dir)]
        if "config" not in cmd and any(["config" in a for a in extra_args]):
            cmd += ["--config", "config/pipeline_config.yml"]
        cmd += extra_args
        print(f"Running: {' '.join(cmd)}")
        if script == "scripts/reconstruct_scene.py":
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    if "[VIS] COLORLESS_VIEW_SECONDS:" in line or "[VIS] COLORED_VIEW_SECONDS:" in line:
                        try:
                            val = float(line.strip().split(":")[-1])
                            view_seconds += val
                        except Exception:
                            pass
                ret = proc.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)
        else:
            subprocess.run(cmd, check=True)

    return view_seconds

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

    start_ts = time.time()
    view_seconds = run_pipeline(project_dir)
    end_ts = time.time()

    # Timing summary (write to file with per-image stats)
    # Count synchronized captures as number of PNGs in RGB dirs (max across sides)
    left_dir = project_dir / "left_camera_rgb"
    right_dir = project_dir / "right_camera_rgb"
    left_count = len(list(left_dir.glob("*.png"))) if left_dir.exists() else 0
    right_count = len(list(right_dir.glob("*.png"))) if right_dir.exists() else 0
    image_count = max(left_count, right_count)

    # Note: start/end measured around entire pipeline run
    total_seconds = end_ts - start_ts
    adjusted_seconds = max(0.0, total_seconds - view_seconds)
    secs_per_capture = (adjusted_seconds / image_count) if image_count > 0 else float('nan')

    summary_lines = [
        f"Project directory: {project_dir}",
        f"Total runtime (s): {total_seconds:.3f}",
        f"Visualization time excluded (s): {view_seconds:.3f}",
        f"Adjusted runtime (s): {adjusted_seconds:.3f}",
        f"Images (max per side): {image_count}",
        f"Seconds per capture (adjusted): {secs_per_capture:.6f}",
        ""
    ]

    print("\n".join(summary_lines))

    out_path = project_dir / "pipeline_runtime.txt"
    try:
        out_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"[Info] Wrote timing summary to: {out_path}")
    except Exception as e:
        print(f"[Warning] Failed to write timing summary: {e}")

if __name__ == "__main__":
    main()
