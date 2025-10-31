import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_confidence_maps(project_dir):
    project_dir = str(project_dir)
    for side in ['left', 'right']:
        subfolder = f"{side}_depth_confidence"
        map_dir = os.path.join(project_dir, subfolder)
        print(f"[Info] Searching for confidence maps in directory: {map_dir}")
        npz_files = sorted(glob.glob(os.path.join(map_dir, "*.npz")))
        print(f"[Info] Found {len(npz_files)} confidence maps in {subfolder}")

        save_dir = os.path.join(map_dir, "confidence_maps_png")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Info] Saving PNGs to: {save_dir}")

        for i, npz_path in enumerate(npz_files, 1):
            print(f"[{side.upper()}][{i}/{len(npz_files)}] Processing: {os.path.basename(npz_path)}")
            data = np.load(npz_path)
            conf = data['confidence_map']
            fname = os.path.basename(npz_path)
            fig, ax = plt.subplots()
            im = ax.imshow(conf, cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f"{side.capitalize()} confidence: {fname}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Confidence")
            ax.axis('off')
            png_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(save_dir, png_name)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
            print(f"[Info] Saved: {out_path}")

    print("[Done] Confidence map visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir",
        type=str,
        required=True,
        help="Path to the project directory containing confidence maps."
    )
    args = parser.parse_args()
    project_dir = args.project_dir
    visualize_confidence_maps(project_dir)