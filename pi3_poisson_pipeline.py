#!/usr/bin/env python3
"""Pi3 → PoissonRecon → SurfaceTrimmer pipeline.

Steps:
  1) Run Pi3 to generate a colored point cloud from images / video.
  2) Estimate oriented normals with Open3D and save an oriented PLY.
  3) Run PoissonRecon to reconstruct a colored mesh with density.
  4) Run SurfaceTrimmer to remove low-density “crap”.

Requirements:
  - Run this from inside the Pi3 repo (so `pi3` is importable).
  - PoissonRecon and SurfaceTrimmer binaries installed / built and on PATH.
  - `pip install open3d torch numpy`.
"""

import logging
import pathlib
import subprocess
import sys

import numpy as np
import open3d as o3d
import torch

import pipeline_config

import pi3.models.pi3 as pi3_model_module
import pi3.utils.basic as pi3_basic
import pi3.utils.geometry as pi3_geom


def run_pi3_to_open3d_cloud(
    data_path, interval, ckpt_path, device_str, conf_thresh, edge_rtol
):
    """Runs Pi3 and returns an Open3D point cloud with colors, no normals yet."""
    # Prepare device.
    device = torch.device(device_str)
    logging.info("Using device: %s", device)

    # Load the Pi3 model.
    if ckpt_path:
        # Load from a local checkpoint.
        logging.info("Loading Pi3 from checkpoint: %s", ckpt_path)
        model = pi3_model_module.Pi3().to(device).eval()
        if ckpt_path.endswith(".safetensors"):
            # Load safetensors weights if needed.
            import safetensors.torch as safetensors_torch  # Imported lazily.

            weights = safetensors_torch.load_file(ckpt_path)
        else:
            # Load a regular PyTorch checkpoint.
            weights = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(weights)
    else:
        # Load from HuggingFace as in the official example.
        logging.info("Loading Pi3 from HuggingFace (yyfz233/Pi3).")
        model = pi3_model_module.Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # Decide interval automatically if negative (same behavior as example.py).
    data_path_str = str(data_path)
    if interval < 0:
        interval = 10 if data_path_str.endswith(".mp4") else 1
    logging.info("Sampling interval: %d", interval)

    # Load images or video frames as tensor: (N, 3, H, W), values in [0, 1].
    logging.info("Loading images from %s", data_path)
    imgs = pi3_basic.load_images_as_tensor(data_path_str, interval=interval).to(device)

    # Run Pi3 inference (with mixed precision on CUDA if available).
    logging.info("Running Pi3 inference...")
    with torch.no_grad():
        if device.type == "cuda":
            major_cap = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if major_cap >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs[None])
        else:
            # CPU path without autocast.
            res = model(imgs[None])

    # Build mask from confidence and depth edges, same logic as example.py.
    conf = torch.sigmoid(res["conf"][..., 0])
    masks = conf > conf_thresh
    non_edge = ~pi3_geom.depth_edge(res["local_points"][..., 2], rtol=edge_rtol)
    masks = torch.logical_and(masks, non_edge)[0]

    # Extract XYZ points and per-point RGB colors from masked locations.
    points = res["points"][0][masks].detach().cpu().numpy()
    colors = imgs.permute(0, 2, 3, 1)[masks].detach().cpu().numpy()

    if points.shape[0] == 0:
        raise RuntimeError("Pi3 produced an empty point cloud after masking.")

    logging.info("Pi3 kept %d points after masking.", points.shape[0])

    # Create Open3D point cloud and assign points & colors.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def estimate_normals_for_cloud(pcd, knn=30, orient_k=50):
    """Estimates oriented normals in-place on an Open3D point cloud."""
    # Guard against empty cloud.
    if len(pcd.points) == 0:
        raise ValueError("Cannot estimate normals on an empty point cloud.")

    # Estimate local normals with a k-NN search.
    logging.info("Estimating normals with k-NN (k=%d)...", knn)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))

    # Orient normals consistently to get an approximately coherent field.
    logging.info("Orienting normals consistently (k=%d)...", orient_k)
    pcd.orient_normals_consistent_tangent_plane(orient_k)


def save_oriented_point_cloud(pcd, ply_path):
    """Saves a point cloud with normals and colors to PLY using Open3D."""
    # Ensure normals exist before saving.
    if not pcd.has_normals():
        raise ValueError(
            "Point cloud has no normals. Call estimate_normals_for_cloud first."
        )

    # Write the point cloud as a PLY file.
    ply_path = pathlib.Path(ply_path)
    logging.info("Writing oriented point cloud to %s", ply_path)
    success = o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
    if not success:
        raise RuntimeError(f"Failed to write point cloud to {ply_path}")


def run_poisson_recon(
    poisson_exe, input_ply, output_ply, depth, samples_per_node, point_weight, threads
):
    """Invokes PoissonRecon to reconstruct a colored mesh with density values.

    Uses:
      --density to store per-vertex 'value' (sampling density) for trimming.
      --colors to propagate input point colors to the mesh vertices.
    """
    # Build command-line for PoissonRecon.
    cmd = [
        poisson_exe,
        "--in",
        str(input_ply),
        "--out",
        str(output_ply),
        "--depth",
        str(depth),
        "--samplesPerNode",
        str(samples_per_node),
        "--pointWeight",
        str(point_weight),
        "--density",
        "--colors",
    ]

    # Optionally set number of threads.
    if threads and threads > 0:
        cmd.extend(["--threads", str(threads)])

    # Run PoissonRecon as a subprocess.
    logging.info("Running PoissonRecon:\n  %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logging.info("PoissonRecon finished. Mesh written to %s", output_ply)


def run_surface_trimmer(
    surface_trimmer_exe, input_ply, output_ply, trim_value, remove_islands, a_ratio
):
    """Invokes SurfaceTrimmer to remove low-density regions and tiny islands.

    Requires that the input mesh has a per-vertex 'value' field, which we
    get by running PoissonRecon with --density.
    """
    # Build command-line for SurfaceTrimmer.
    cmd = [
        surface_trimmer_exe,
        "--in",
        str(input_ply),
        "--out",
        str(output_ply),
        "--trim",
        str(trim_value),
        "--aRatio",
        str(a_ratio),
    ]

    # Optionally remove isolated small connected components.
    if remove_islands:
        cmd.append("--removeIslands")

    # Run SurfaceTrimmer as a subprocess.
    logging.info("Running SurfaceTrimmer:\n  %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logging.info("SurfaceTrimmer finished. Trimmed mesh written to %s", output_ply)


def parse_args(argv):
    """Parses command-line arguments for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pi3 → PoissonRecon → SurfaceTrimmer pipeline"
    )

    # Add common arguments from shared configuration
    pipeline_config.add_common_args(parser)

    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for the pipeline script."""
    # Parse command-line arguments.
    args = parse_args(argv or sys.argv[1:])

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Prepare output paths.
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oriented_points_path = output_dir / f"{args.output_name}_points_oriented.ply"
    poisson_mesh_path = output_dir / f"{args.output_name}_poisson_mesh.ply"
    trimmed_mesh_path = output_dir / f"{args.output_name}_trimmed_mesh.ply"

    # Step 1: Run Pi3 to obtain a colored point cloud (no normals yet).
    logging.info("=== Step 1: Pi3 → colored point cloud ===")
    pcd = run_pi3_to_open3d_cloud(
        data_path=pathlib.Path(args.data_path),
        interval=args.interval,
        ckpt_path=args.ckpt,
        device_str=args.device,
        conf_thresh=args.conf_thresh,
        edge_rtol=args.edge_rtol,
    )

    # Step 2: Estimate normals and save oriented point cloud.
    logging.info("=== Step 2: Estimate normals and save oriented PLY ===")
    estimate_normals_for_cloud(pcd)
    save_oriented_point_cloud(pcd, oriented_points_path)

    # Step 3: Run Poisson surface reconstruction (mesh + density + vertex colors).
    logging.info("=== Step 3: PoissonRecon → colored mesh with density ===")
    run_poisson_recon(
        poisson_exe=args.poisson_exe,
        input_ply=oriented_points_path,
        output_ply=poisson_mesh_path,
        depth=args.depth,
        samples_per_node=args.samples_per_node,
        point_weight=args.point_weight,
        threads=args.threads,
    )

    # Step 4: Run SurfaceTrimmer to remove low-density crap / islands.
    logging.info("=== Step 4: SurfaceTrimmer → trimmed mesh ===")
    run_surface_trimmer(
        surface_trimmer_exe=args.surface_trimmer_exe,
        input_ply=poisson_mesh_path,
        output_ply=trimmed_mesh_path,
        trim_value=args.trim,
        remove_islands=not args.no_remove_islands,
        a_ratio=args.a_ratio,
    )

    # Optionally clean up intermediate files.
    if not args.keep_intermediate:
        logging.info("Removing intermediate files.")
        try:
            oriented_points_path.unlink(missing_ok=True)
            poisson_mesh_path.unlink(missing_ok=True)
        except TypeError:
            # Python <3.8 compatibility: missing_ok not available.
            for p in (oriented_points_path, poisson_mesh_path):
                if p.exists():
                    p.unlink()

    logging.info("Pipeline complete.")
    logging.info("Final trimmed mesh: %s", trimmed_mesh_path)


if __name__ == "__main__":
    main()
