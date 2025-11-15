#!/usr/bin/env python3
import numpy as np
import open3d as o3d

def add_normals(in_path: str, out_path: str) -> None:
    # Load point cloud from PLY
    pcd = o3d.io.read_point_cloud(in_path)

    # Estimate normals from k-NN neighborhood
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=32)
    )

    # Make normals roughly consistent (orient them coherently)
    pcd.orient_normals_consistent_tangent_plane(k=64)

    # Save back to PLY (with normals)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)

if __name__ == "__main__":
    add_normals("save2.ply", "kitchen2_with_normals.ply")
