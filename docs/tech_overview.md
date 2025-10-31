## Meta Quest 3D Reconstruction – Technical Overview

This document describes the system architecture, processing stages, configuration, data models, and outputs for the 3D reconstruction pipeline.

### Goals
- Reconstruct a scene from Meta Quest passthrough color and depth streams
- Provide a volumetric Open3D-based pipeline and a COLMAP export flow
- Enable selective/cached execution, QA visualization, and downstream SfM usage

### High-level Architecture
- **Entry points**
  - `scripts/run_full_pipeline.py`: CLI orchestration invoking each stage as a subprocess
  - `scripts/reconstruct_scene.py`: CLI wrapper around `PipelineProcessor`
  - `scripts/build_colmap_project.py`: COLMAP export utility
- **Core orchestrator**
  - `scripts/pipeline/pipeline_processor.py` wires IO, config parsing, and stages
- **Processing modules**
  - `scripts/processing/yuv_conversion/` – YUV420_888 → BGR (RGB)
  - `scripts/processing/depth_conversion/` – depth visualization to linear map (QA)
  - `scripts/processing/reconstruction/` – confidence estimation, pose optimization, TSDF integration, color map optimization
- **Data IO**
  - `scripts/dataio/` and `scripts/config/project_path_config.py` define datasets and filesystem layout
- **Configuration**
  - `config/pipeline_config.yml` controls all tunables and stage toggles

### Orchestration and Entry Points

```13:33:scripts/run_full_pipeline.py
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
```

```1:24:scripts/pipeline/pipeline_processor.py
from pathlib import Path
from config.pipeline_configs import PipelineConfigs
from dataio.data_io import DataIO
from processing.depth_conversion.convert_depth_to_linear import convert_depth_directory
from processing.reconstruction.reconstruct_scene import reconstruct_scene
from processing.yuv_conversion.convert_yuv_dir import convert_yuv_directory


class PipelineProcessor:
    def __init__(self, project_dir: Path, config_yml_path: Path):
        self.data_io = DataIO(project_dir=project_dir)
        self.pipeline_configs = PipelineConfigs.parse_config_yml(config_yml_path)


    def convert_yuv_to_rgb(self):
        convert_yuv_directory(image_io=self.data_io.color, config=self.pipeline_configs.yuv_to_rgb)
    
    def convert_depth_to_linear(self):
        convert_depth_directory(depth_data_io=self.data_io.depth, depth_to_linear_config=self.pipeline_configs.depth_to_linear)


    def reconstruct_scene(self):
        reconstruct_scene(data_io=self.data_io, config=self.pipeline_configs.reconstruction)
```

### Configuration
All stage behavior is configured via `config/pipeline_config.yml`.

```1:31:config/pipeline_config.yml
yuv_to_rgb:
  blur_filter: False
  blur_threshold: 50.0
  exposure_filter: False
  exposure_threshold_low: 0.05
  exposure_threshold_high: 0.05

depth_to_linear:
  clip_near_m: 0.1
  clip_far_m: 3.0
  use_cache: True

reconstruction:
  device: "CPU:0"
  use_dataset_cache: true
  estimate_depth_confidences: true
  optimize_depth_pose: true
  use_fragment_dataset_cache: true
  use_optimized_dataset_cache: true
  use_colorless_vbg_cache: true
  visualize_colorless_pcd: true
  optimize_color_pose: true
  visualize_colored_mesh: true
  sample_point_cloud_from_colored_mesh: true
  points_per_vertex_ratio: 2.0
  render_color_aligned_depth: true

  confidence_estimation:
    target_frame_range: 10
```

### Data Models and IO
- `CameraDataset` and `DepthDataset` represent per-frame timestamps, intrinsics, extrinsics, and depth metadata.
- `DataIO` composes `ImageDataIO`, `DepthDataIO`, and `ReconstructionDataIO` with consistent paths from `project_path_config`.
- Reconstruction artifacts and caches are read/written via `ReconstructionDataIO`:

```9:41:scripts/dataio/reconstruction_data_io.py
class ReconstructionDataIO:
    def __init__(self, reconstruction_path_config: ReconstructionPathConfig):
        
    
    def load_fragment_datasets(self) -> dict[Side, list[DepthDataset]]:
        fragment_path_map = self.reconstruction_path_config.get_fragment_dataset_paths()
        fragment_datasets: dict[Side, list[DepthDataset]] = {}
        for side, paths in fragment_path_map.items():
            fragment_datasets[side] = [DepthDataset.load(path) for path in paths]
        return fragment_datasets    
    
    def save_fragment_dataset(self, dataset: DepthDataset, side: Side, index: int):
        path = self.reconstruction_path_config.get_fragment_dataset_path(side=side, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(path)
    
    def load_fragment_pcd(self, side: Side, index: int) -> o3d.t.geometry.PointCloud:
        path = self.reconstruction_path_config.get_fragment_pcd_path(side=side, index=index)
        return o3d.t.io.read_point_cloud(str(path))
    
    def save_fragment_pcd(self, pcd: o3d.t.geometry.PointCloud, side: Side, index: int):
        path = self.reconstruction_path_config.get_fragment_pcd_path(side=side, index=index)
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.t.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)
```

```42:79:scripts/dataio/reconstruction_data_io.py
def load_colorless_vbg(self) -> Optional[o3d.t.geometry.VoxelBlockGrid]:
        colorless_vbg_path = self.reconstruction_path_config.get_colorless_vbg_path()
        if not colorless_vbg_path.exists():
            
        return o3d.t.geometry.VoxelBlockGrid.load(str(colorless_vbg_path))
    
    def save_colorless_vbg(self, vbg: o3d.t.geometry.VoxelBlockGrid):
        colorless_vbg_path = self.reconstruction_path_config.get_colorless_vbg_path()
        colorless_vbg_path.parent.mkdir(parents=True, exist_ok=True)
        vbg.save(str(colorless_vbg_path))

    def load_colored_mesh(self) -> Optional[o3d.t.geometry.TriangleMesh]:
        color_mesh_path = self.reconstruction_path_config.get_colored_mesh_path()
        if not color_mesh_path.exists():
            return None
        
        return o3d.t.io.read_triangle_mesh(str(color_mesh_path))
    
    def save_colored_mesh_legacy(self, mesh: o3d.geometry.TriangleMesh):
        color_mesh_path = self.reconstruction_path_config.get_colored_mesh_path()
        color_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(color_mesh_path), mesh)
    
    def save_colored_mesh(self, mesh: o3d.t.geometry.TriangleMesh):
        color_mesh_path = self.reconstruction_path_config.get_colored_mesh_path()
        color_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.t.io.write_triangle_mesh(str(color_mesh_path), mesh)
```

### Processing Stages

#### 1) YUV → RGB Conversion (optional filtering)
- Converts Quest YUV420_888 images to BGR, applies optional blur/exposure filters.
- Outputs RGB images per side.

```13:33:scripts/processing/yuv_conversion/convert_yuv_dir.py
def process_file(
    side: Side,
    timestamp: int,
    image_io: ImageDataIO,
    filter: Optional[Callable[[np.ndarray], bool]] = None,
) -> bool:
    raw_data = image_io.load_yuv(side=side, timestamp=timestamp)
    format_info = image_io.load_image_format_info(side=side)
    bgr_img = convert_yuv420_888_to_bgr(raw_data=raw_data, format_info=format_info)
    if filter is not None and not filter(bgr_img):
        return False
    image_io.save_bgr(bgr=bgr_img, side=side, timestamp=timestamp)
    return True
```

#### 2) Depth → Linear Map (QA visualization)
- Produces 8-bit linearized depth images for QA using near/far clipping.

```8:46:scripts/processing/depth_conversion/convert_depth_to_linear.py
def convert_depth_directory(
    depth_data_io: DepthDataIO,
    depth_to_linear_config: Depth2LinearConfig
):
    ...
    linear_depth_map = np.clip((depth_map - clip_near) / (clip_far - clip_near), 0, 1) * 255.0
    depth_data_io.save_linear_depth_map(depth_map=linear_depth_map, side=side, timestamp=timestamp)
```

#### 3) Depth Confidence Estimation
- Computes per-pixel confidence by multi-view geometric consistency across neighboring frames.

```115:149:scripts/processing/reconstruction/confidence_estimation/estimate_depth_confidences.py
def estimate_depth_confidences(...):
    dataset = depth_data_io.load_depth_dataset(side=side)
    intrinsic_matrices = compute_o3d_intrinsic_matrices(dataset=dataset)
    extrinsic_matrices = dataset.transforms.convert_coordinate_system(
        target_coordinate_system=CoordinateSystem.OPEN3D,
        is_camera=True
    ).extrinsics_cw
    extrinsic_matrices_inv = np.linalg.inv(extrinsic_matrices)
    parallel_map(build_and_save_confidence_map, args_list=..., use_multiprocessing=config.use_multi_threading)
```

Core error computation:

```73:143:scripts/processing/reconstruction/confidence_estimation/compute_pixel_error_map.py
def compute_pixel_error_map(...):
    # back-project ref, transform to target, project and sample target, back-project and compare
    confidence_map = np.full_like(ref_depth_map, fill_value=np.nan, dtype=np.float32)
    return confidence_map
```

#### 4) Depth Pose Optimization (fragments + pose graph)
- Splits sequences into fragments, aligns locally, then builds and refines a global pose graph to improve depth camera poses.
- Controlled by `reconstruction.optimize_depth_pose` and caching flags.
- See `scripts/processing/reconstruction/depth_optimization/` (e.g., `depth_pose_optimizer.py`, `make_fragments.py`, `refine_fragment_poses.py`).

#### 5) TSDF Integration (Open3D VoxelBlockGrid)
- Integrates depth maps (optionally confidence-filtered) into a TSDF voxel grid.
- Saves colorless TSDF and can visualize a point cloud extraction.

```55:79:scripts/processing/reconstruction/reconstruct_scene.py
if vbg is None:
    log_step("Integrate depth maps")
    for side, dataset in depth_dataset_map.items():
        vbg = integrate(
            dataset=dataset,
            depth_data_io=data_io.depth,
            side=side,
            use_confidence_filtered_depth=integration_config.use_confidence_filtered_depth,
            confidence_threshold=integration_config.confidence_threshold,
            valid_count_threshold=integration_config.valid_count_threshold,
            voxel_size=integration_config.voxel_size,
            block_resolution=integration_config.block_resolution,
            block_count=integration_config.block_count,
            depth_max=integration_config.depth_max,
            trunc_voxel_multiplier=integration_config.trunc_voxel_multiplier,
            device=integration_config.device,
            show_progress=True,
            desc=f"[{side.name}] Integrating depth maps ...",
            vbg_opt=vbg
        )
```

#### 6) Color Pose Optimization and Mesh Colorization
- Raycasts synthetic depths from TSDF for color frames, builds RGBD set and camera trajectory, and runs Open3D color map rigid optimizer.
- Saves colored mesh and optionally samples a dense colored point cloud.

```24:71:scripts/processing/reconstruction/color_map_optimization/optimize_color_pose.py
for side in Side:
    color_dataset = data_io.color.load_color_dataset(side=side, use_cache=config.use_dataset_cache)
    color_dataset = color_dataset[::config.interval]
    color_dataset.transforms = color_dataset.transforms.convert_coordinate_system(
        target_coordinate_system=CoordinateSystem.OPEN3D,
        is_camera=True
    )
    depth_map_iter = raycast_in_color_view(scene=scene, dataset=color_dataset)
    ...
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as _:
    colored_mesh, trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
        mesh.cpu().to_legacy(), rgbd_images, trajectory,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=config.max_iteration)
    )
```

### Outputs and Formats
- Preprocessing
  - `left_camera_rgb/`, `right_camera_rgb/`: PNG/JPG RGB images
  - `*_linear_depth/`: 8-bit QA depth previews
- Reconstruction
  - Colorless TSDF `VoxelBlockGrid`: saved via `vbg.save(...)`
  - Colored mesh: written via Open3D (`.ply`/tensor mesh)
  - Colored point cloud (optional): sampled from colored mesh
  - Confidence maps: per-frame float32 images/maps
  - Optimized datasets: serialized `CameraDataset`/`DepthDataset` artifacts
- COLMAP export (`scripts/build_colmap_project.py`)
  - `images/`: RGB copies (interval downsampling)
  - `distorted/sparse/0/`: `cameras.bin`, `images.bin`, `points3D.bin`

```191:224:scripts/build_colmap_project.py
def main(args):
    data_io = DataIO(project_dir=args.project_dir)
    dataset_map = load_dataset_map(
        data_io=data_io,
        use_optimized_color_dataset=args.use_optimized_color_dataset
    )
    model_dir = args.output_dir / "distorted/sparse/0"
    input_dir = args.output_dir / "images"
    cameras, images = read_cameras_and_images(
        data_io=data_io,
        dataset_map=dataset_map,
        input_dir=input_dir,
        interval=args.interval
    )
    points3d = read_points_3d(data_io=data_io) if args.use_colored_pointcloud else {}
    write_model(cameras=cameras, images=images, points3D=points3d, path=model_dir, ext=".bin")
```

### Performance and Device Notes
- Open3D tensor ops are driven by `reconstruction.device` (e.g., `CPU:0` on macOS).
- Confidence estimation uses parallelization (threads or processes) and batching.
- Caching toggles (`use_*_cache`) avoid recomputation for large sequences.

### Extensibility Guidelines
- Add new stages under `scripts/processing/...` with a dedicated config dataclass and YAML section.
- Register the stage in `PipelineProcessor` and (optionally) the full runner.
- Keep coordinate frames consistent: convert to `OPEN3D` before using Open3D APIs.
- Extend `project_path_config.py` and `DataIO` for any new on-disk artifacts.

### Typical End-to-End Flow
1. Convert YUV → RGB, optionally filter low-quality frames
2. Generate depth linear previews (QA)
3. Load/build color and depth datasets
4. Estimate depth confidences
5. Optimize depth poses (optional)
6. TSDF integrate depth into colorless VoxelBlockGrid
7. Visualize point cloud (optional)
8. Optimize color poses and colorize mesh; optionally sample dense colored point cloud
9. Export COLMAP project for SfM pipelines
