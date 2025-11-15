"""Shared configuration for Pi3 Poisson pipeline parameters."""

import argparse


def add_common_args(parser):
    """Add common command-line arguments shared by both scripts."""

    # Input / Pi3 args.
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input image directory or video file (mp4).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=-1,
        help="Frame sampling interval; <0 means auto (1 for images, 10 for video).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional Pi3 checkpoint path (.safetensors or .pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string, e.g. 'cuda' or 'cpu'.",
    )

    # Output / paths.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where all outputs will be saved.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="scene",
        help="Base name used for generated files.",
    )

    # PoissonRecon args.
    parser.add_argument(
        "--poisson_exe",
        type=str,
        default="PoissonRecon",
        help="Path or name of PoissonRecon executable.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=11,
        help="Poisson reconstruction depth (octree depth).",
    )
    parser.add_argument(
        "--samples_per_node",
        type=float,
        default=1.5,
        help="PoissonRecon --samplesPerNode parameter.",
    )
    parser.add_argument(
        "--point_weight",
        type=float,
        default=2.0,
        help="PoissonRecon --pointWeight (screening strength).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads for PoissonRecon; 0 = default (all cores).",
    )

    # SurfaceTrimmer args.
    parser.add_argument(
        "--surface_trimmer_exe",
        type=str,
        default="SurfaceTrimmer",
        help="Path or name of SurfaceTrimmer executable.",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=7.0,
        help="SurfaceTrimmer --trim threshold (typical values ~5–10). ",
    )
    parser.add_argument(
        "--a_ratio",
        type=float,
        default=0.001,
        help="SurfaceTrimmer --aRatio (island area ratio).",
    )
    parser.add_argument(
        "--no_remove_islands",
        action="store_true",
        help="If set, do NOT pass --removeIslands to SurfaceTrimmer.",
    )

    # Pi3 filtering args.
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.1,
        help="Confidence threshold on Pi3 points (before edge filtering).",
    )
    parser.add_argument(
        "--edge_rtol",
        type=float,
        default=0.03,
        help="Relative tolerance used in depth_edge() for edge masking.",
    )

    # Misc.
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate oriented point cloud and raw Poisson mesh.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )


def get_common_args_dict(args):
    """Extract common arguments from parsed args as a dictionary."""
    common_attrs = [
        'data_path', 'interval', 'ckpt', 'device', 'output_dir', 'output_name',
        'poisson_exe', 'depth', 'samples_per_node', 'point_weight', 'threads',
        'surface_trimmer_exe', 'trim', 'a_ratio', 'no_remove_islands',
        'conf_thresh', 'edge_rtol', 'keep_intermediate', 'log_level'
    ]
    return {attr: getattr(args, attr) for attr in common_attrs if hasattr(args, attr)}