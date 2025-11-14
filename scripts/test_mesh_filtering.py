#!/usr/bin/env python3
"""
Test script to verify mesh filtering logic
"""
import numpy as np
import open3d as o3d

def filter_mesh_components(
    mesh: o3d.t.geometry.TriangleMesh,
    min_triangle_count: int = 2000
) -> o3d.t.geometry.TriangleMesh:
    """
    Filter out disconnected mesh components that have fewer triangles than the threshold.
    """
    # Convert to legacy format for connected component analysis
    mesh_legacy = mesh.to_legacy()
    
    # Find connected components
    triangle_clusters, cluster_n_triangles, cluster_area = mesh_legacy.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    print(f"[Debug] Found {len(cluster_n_triangles)} connected components")
    print(f"[Debug] Component sizes: {cluster_n_triangles}")
    print(f"[Debug] Threshold: {min_triangle_count}")
    
    # Create mask to keep only components with triangle count >= min_triangle_count
    valid_clusters = np.where(cluster_n_triangles >= min_triangle_count)[0]
    mask = np.isin(triangle_clusters, valid_clusters)
    
    print(f"[Debug] Valid clusters (>= {min_triangle_count}): {valid_clusters}")
    print(f"[Debug] Triangles to keep: {np.sum(mask)} / {len(mask)}")
    
    # Remove triangles from small components
    mesh_legacy.remove_triangles_by_mask(~mask)
    mesh_legacy.remove_unreferenced_vertices()
    
    # Clean up degenerate triangles
    mesh_legacy.remove_degenerate_triangles()
    mesh_legacy.remove_duplicated_triangles()
    mesh_legacy.remove_duplicated_vertices()
    mesh_legacy.remove_non_manifold_edges()
    
    # Convert back to tensor format
    filtered_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    
    # Preserve device
    filtered_mesh = filtered_mesh.to(mesh.device)
    
    num_removed = len(cluster_n_triangles) - len(valid_clusters)
    if num_removed > 0:
        print(f"[Info] Removed {num_removed} small mesh component(s) with < {min_triangle_count} triangles")
        print(f"[Info] Kept {len(valid_clusters)} component(s) with >= {min_triangle_count} triangles")
    else:
        print(f"[Info] No components removed (all components have >= {min_triangle_count} triangles)")
    
    return filtered_mesh

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: test_mesh_filtering.py <mesh_file.ply> [threshold]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    
    print(f"[Info] Loading mesh from {mesh_path}")
    mesh = o3d.t.io.read_triangle_mesh(mesh_path)
    print(f"[Info] Original mesh has {len(mesh.triangles)} triangles")
    
    filtered = filter_mesh_components(mesh, min_triangle_count=threshold)
    print(f"[Info] Filtered mesh has {len(filtered.triangles)} triangles")
    
    output_path = mesh_path.replace(".ply", "_filtered.ply")
    o3d.t.io.write_triangle_mesh(output_path, filtered)
    print(f"[Info] Saved filtered mesh to {output_path}")

