import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import sys
from pathlib import Path as _P

# Ensure 'scripts' package root on sys.path (to mirror other scene scripts)
_SCRIPTS_ROOT = _P(__file__).resolve().parents[2]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from config.pipeline_configs import PipelineConfigs
from dataio.data_io import DataIO
from models.side import Side
from processing.reconstruction.utils.o3d_utils import compute_o3d_intrinsic_matrices


def axis_vector(name: str) -> np.ndarray:
    n = name.lower()
    if n == 'x':
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if n == 'y':
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def world_transform_for_turntable(angle_rad: float, axis_name: str, pivot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    ax = axis_vector(axis_name)
    Rm = R.from_rotvec(ax * angle_rad).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    # rotation about pivot: p' = R*(p - pivot) + pivot
    T[:3, 3] = pivot - Rm @ pivot
    return T


def load_masked_depth_npy(root: Path, side: Side, timestamp: int) -> np.ndarray | None:
    ddir = root / ("left_depth_linear_masked" if side == Side.LEFT else "right_depth_linear_masked")
    f = ddir / f"{timestamp}.npy"
    if not f.exists():
        return None
    try:
        return np.load(f)
    except Exception:
        return None


def fuse_side(project_dir: Path, side: Side, device: o3d.core.Device, recon_cfg,
              rpm: float | None, start_time_ms: int | None, axis_name: str, pivot: np.ndarray) -> o3d.t.geometry.VoxelBlockGrid:
    data_io = DataIO(project_dir=project_dir)
    depth_ds = data_io.depth.load_depth_dataset(side=side, use_cache=True)

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1)),
        voxel_size=float(recon_cfg.depth_integration.voxel_size),
        block_resolution=int(recon_cfg.depth_integration.block_resolution),
        block_count=int(recon_cfg.depth_integration.block_count),
        device=device,
    )

    extrinsics_wc = depth_ds.transforms.extrinsics_wc
    intrinsics = compute_o3d_intrinsic_matrices(depth_ds)

    for i in range(len(depth_ds.timestamps)):
        ts = int(depth_ds.timestamps[i])
        depth_m = load_masked_depth_npy(project_dir, side, ts)
        if depth_m is None:
            continue
        depth_img = o3d.t.geometry.Image(o3d.core.Tensor(depth_m.astype(np.float32), dtype=o3d.core.Dtype.Float32, device=device))

        intrinsic = o3d.core.Tensor(intrinsics[i], dtype=o3d.core.Dtype.Float64)

        # Optional object-rotation compensation
        ext_cw = extrinsics_wc[i]
        if rpm is not None and start_time_ms is not None:
            # angle = omega * dt, omega = 2*pi*rpm/60
            dt_s = max(0.0, (ts - start_time_ms) / 1000.0)
            omega = 2.0 * np.pi * float(rpm) / 60.0
            angle = omega * dt_s
            # World is rotated by R(angle) due to object spin. To stabilize object, apply W_inv to camera-to-world.
            W = world_transform_for_turntable(angle, axis_name=axis_name, pivot=pivot)
            Winv = np.linalg.inv(W)
            ext_cw = Winv @ ext_cw

        extrinsic = o3d.core.Tensor(ext_cw, dtype=o3d.core.Dtype.Float64)

        coords = vbg.compute_unique_block_coordinates(
            depth=depth_img,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_scale=1.0,
            depth_max=float(recon_cfg.depth_integration.depth_max),
            trunc_voxel_multiplier=float(recon_cfg.depth_integration.trunc_voxel_multiplier),
        )
        vbg.integrate(
            block_coords=coords,
            depth=depth_img,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_scale=1.0,
            depth_max=float(recon_cfg.depth_integration.depth_max),
            trunc_voxel_multiplier=float(recon_cfg.depth_integration.trunc_voxel_multiplier),
        )

    return vbg


def main():
    ap = argparse.ArgumentParser(description="Fuse masked depth maps into a TSDF (object-centric)")
    ap.add_argument("--project_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sides", type=str, default="left,right")
    ap.add_argument("--rpm", type=float, default=None, help="Turntable RPM (constant)")
    ap.add_argument("--start_time_ms", type=int, default=None, help="Timestamp (ms) when rotation starts")
    ap.add_argument("--axis", type=str, default="z", choices=["x","y","z"], help="Turntable axis in world frame")
    ap.add_argument("--pivot", type=str, default=None, help="Pivot as 'x,y,z' in meters (world coords). Defaults to origin")
    args = ap.parse_args()

    cfg = PipelineConfigs.parse_config_yml(args.config)
    device = cfg.reconstruction.device

    wanted = [s.strip() for s in args.sides.split(',') if s.strip()]
    sides = []
    if wanted == ["left"]:
        sides = [Side.LEFT]
    elif wanted == ["right"]:
        sides = [Side.RIGHT]
    else:
        sides = [Side.LEFT, Side.RIGHT]

    # Parse pivot
    if args.pivot:
        pv = np.array([float(v) for v in args.pivot.split(',')], dtype=np.float64)
    else:
        pv = np.zeros(3, dtype=np.float64)

    vbgs = []
    for s in sides:
        vbgs.append(fuse_side(args.project_dir, s, device, cfg.reconstruction,
                              rpm=args.rpm, start_time_ms=args.start_time_ms,
                              axis_name=args.axis, pivot=pv))

    # Merge volumes by simple re-integration (optional). For now, just extract mesh from the first available.
    for idx, vbg in enumerate(vbgs):
        if vbg is None:
            continue
        mesh = vbg.extract_triangle_mesh(weight_threshold=cfg.reconstruction.color_aligned_depth_rendering.weight_threshold)
        out_path = args.project_dir / f"reconstruction/object_mesh_{sides[idx].name.lower()}.ply"
        o3d.t.io.write_triangle_mesh(str(out_path), mesh)
        print(f"[Info] Wrote mesh: {out_path}")


if __name__ == "__main__":
    main()


