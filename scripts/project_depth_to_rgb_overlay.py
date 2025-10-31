import os
import argparse
import cv2
import numpy as np
import json

# Optionally, use Open3D for visualization if available
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

def parse_args():
    parser = argparse.ArgumentParser(description="Project depth data as point cloud overlay onto RGB images with support for separate RGB/depth intrinsics and extrinsic transform.")
    # Input folders
    parser.add_argument('--left_rgb', required=True, help="Path to left RGB images directory")
    parser.add_argument('--left_depth', required=True, help="Path to left depth images directory")
    parser.add_argument('--right_rgb', required=True, help="Path to right RGB images directory")
    parser.add_argument('--right_depth', required=True, help="Path to right depth images directory")
    parser.add_argument('--output', required=True, help="Output directory for overlayed images")
    parser.add_argument('--visualize', action='store_true', help="Visualize instead of saving overlays (uses Open3D if available)")

    # Camera intrinsics (RGB)
    parser.add_argument('--left_rgb_intrinsics', type=str, required=False, help='Left RGB camera intrinsics JSON path')
    parser.add_argument('--right_rgb_intrinsics', type=str, required=False, help='Right RGB camera intrinsics JSON path')
    # Camera intrinsics (depth)
    parser.add_argument('--left_depth_intrinsics', type=str, required=False, help='Left depth camera intrinsics JSON path')
    parser.add_argument('--right_depth_intrinsics', type=str, required=False, help='Right depth camera intrinsics JSON path')

    # Extrinsics (depth->rgb)
    parser.add_argument('--left_extrinsics', type=str, required=False, help='Left depth-to-RGB extrinsic transform (JSON, 4x4 matrix or T+Q)')
    parser.add_argument('--right_extrinsics', type=str, required=False, help='Right depth-to-RGB extrinsic transform (JSON, 4x4 matrix or T+Q)')
    
    return parser.parse_args()

def load_image_pairs(rgb_dir, depth_dir):
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if any(f.lower().endswith(ext) for ext in ['.jpg', '.png'])])
    depth_files = sorted([f for f in os.listdir(depth_dir) if any(f.lower().endswith(ext) for ext in ['.png', '.npy'])])
    pairs = [(os.path.join(rgb_dir, f), os.path.join(depth_dir, f)) for f in rgb_files if f in depth_files]
    return pairs

def load_intrinsics(json_path):
    # JSON file with {"intrinsics": {"fx": float, "fy": float, "cx": float, "cy": float}}
    with open(json_path, 'r') as f:
        d = json.load(f)
    intr = d["intrinsics"] if "intrinsics" in d else d
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    return fx, fy, cx, cy

def load_extrinsics(json_path):
    # Either {"matrix": [[4x4 values]]} or {"translation": [x, y, z], "rotation": [qx, qy, qz, qw]}
    with open(json_path, 'r') as f:
        d = json.load(f)
    if "matrix" in d:
        extr = np.array(d["matrix"], dtype=np.float32)
    elif "translation" in d and "rotation" in d:
        extr = make_extrinsic_matrix(d["translation"], d["rotation"])
    else:
        raise ValueError(f"Invalid extrinsics format in {json_path}")
    return extr

def make_extrinsic_matrix(translation, quat):
    # translation: [x, y, z]; quat: [qx, qy, qz, qw] (xyzw order)
    t = np.array(translation, dtype=np.float32)
    q = np.array(quat, dtype=np.float32)
    # Quaternion to rotation
    q = q / np.linalg.norm(q)
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x**2-2*y**2]
    ], dtype=np.float32)
    extr = np.eye(4, dtype=np.float32)
    extr[:3,:3] = R
    extr[:3,3]  = t
    return extr

def transform_point_cloud(pts, extr):
    # pts: (N,3), extr: (4,4)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    pts_trans = (extr @ pts_h.T).T[:, :3]
    return pts_trans

def depth_to_point_cloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    pts = np.stack((x, y, z), axis=-1)
    mask = (z > 0)
    return pts[mask]

def overlay_point_cloud_on_image(rgb, points_2d):
    overlay = rgb.copy()
    for pt in points_2d:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]:
            cv2.circle(overlay, (x, y), 1, (0,255,0), -1)
    return overlay

def project_to_image(points, fx, fy, cx, cy):
    # Project 3D points to image plane
    x = (points[:,0] * fx / points[:,2]) + cx
    y = (points[:,1] * fy / points[:,2]) + cy
    return np.vstack((x, y)).T

def process_side(side, rgb_dir, depth_dir, out_dir, rgb_intrin, depth_intrin, extr=None, visualize=False):
    pairs = load_image_pairs(rgb_dir, depth_dir)
    fx_d, fy_d, cx_d, cy_d = depth_intrin
    fx_rgb, fy_rgb, cx_rgb, cy_rgb = rgb_intrin
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for rgb_path, depth_path in pairs:
        rgb_img = cv2.imread(rgb_path)
        if depth_path.lower().endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        points_3d = depth_to_point_cloud(depth, fx_d, fy_d, cx_d, cy_d)
        if extr is not None:
            points_3d = transform_point_cloud(points_3d, extr)
        pts_2d = project_to_image(points_3d, fx_rgb, fy_rgb, cx_rgb, cy_rgb)
        overlay = overlay_point_cloud_on_image(rgb_img, pts_2d)
        out_path = os.path.join(out_dir, os.path.basename(rgb_path))
        if visualize and HAS_O3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            o3d.visualization.draw_geometries([pcd])
        else:
            cv2.imwrite(out_path, overlay)
            print(f"Saved overlay for {side}: {out_path}")

def main():
    args = parse_args()

    def get_intrinsics(path, side_name, default):
        if path:
            return load_intrinsics(path)
        else:
            print(f"Warning: Using default {side_name} intrinsics (fx, fy, cx, cy) = {default}")
            return default
    def get_extrinsics(path, side_name):
        if path:
            return load_extrinsics(path)
        else:
            print(f"No extrinsic transform provided for {side_name}, assuming pre-aligned")
            return None

    default_rgb_left = [868.0860595703125, 868.0860595703125, 641.6399536132813, 482.6952209472656]
    default_rgb_right = [869.45703125, 869.45703125, 636.903076171875, 477.7011413574219]
    # These should be set to your depth camera intrinsics (currently assume identical to RGB for simplicity)
    default_depth = [869.45703125, 869.45703125, 636.903076171875, 477.7011413574219]

    left_rgb_intrin = get_intrinsics(args.left_rgb_intrinsics, "left RGB", default_rgb_left)
    right_rgb_intrin = get_intrinsics(args.right_rgb_intrinsics, "right RGB", default_rgb_right)
    left_depth_intrin = get_intrinsics(args.left_depth_intrinsics, "left depth", default_depth)
    right_depth_intrin = get_intrinsics(args.right_depth_intrinsics, "right depth", default_depth)
    left_extr = get_extrinsics(args.left_extrinsics, "left")
    right_extr = get_extrinsics(args.right_extrinsics, "right")

    process_side('left', args.left_rgb, args.left_depth, os.path.join(args.output, 'left'), left_rgb_intrin, left_depth_intrin, left_extr, args.visualize)
    process_side('right', args.right_rgb, args.right_depth, os.path.join(args.output, 'right'), right_rgb_intrin, right_depth_intrin, right_extr, args.visualize)

if __name__ == "__main__":
    main()
