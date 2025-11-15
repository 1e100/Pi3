#!/usr/bin/env python3
"""
Script to run the Pi3 Poisson pipeline Docker container with proper file mappings and flags.

This script handles:
- Building the Docker image (if needed)
- Mounting input/output directories
- Passing command-line arguments to the container
- Managing GPU access when available
"""

import argparse
import pathlib
import subprocess
import sys
import time

import pipeline_config

CONTAINER_NAME = "pi3-poisson:latest"


def build_container_args(args):
    """Build command-line arguments for the container from parsed args."""
    container_args = []

    # Get common pipeline arguments as dictionary
    common_args = pipeline_config.get_common_args_dict(args)

    # Convert arguments to command-line format for the container
    for arg_name, arg_value in common_args.items():
        if arg_value is None or arg_value is False:
            continue  # Skip None and False values

        # Skip data_path and output_dir as we'll handle them separately
        if arg_name in ['data_path', 'output_dir']:
            continue

        if isinstance(arg_value, bool) and arg_value:
            # For boolean flags that are True
            container_args.append(f"--{arg_name}")
        elif not isinstance(arg_value, bool):
            # For arguments with values
            container_args.extend([f"--{arg_name}", str(arg_value)])

    return container_args


def run_docker_container(image_name, args, container_name=None):
    """Run the Docker container with the specified arguments."""

    # Convert paths to absolute paths
    data_path = pathlib.Path(args.data_path).absolute()
    output_dir = pathlib.Path(args.output_dir).absolute()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare container paths
    container_data_path = "/input/data"
    container_output_path = "/output"

    # Build docker run command
    cmd = ["docker", "run", "--gpus", "all"]

    if container_name:
        cmd.extend(["--name", container_name])

    # Add volume mounts
    cmd.extend(
        [
            "-v",
            f"{data_path}:{container_data_path}:ro",  # Read-only input
            "-v",
            f"{output_dir}:{container_output_path}",  # Read-write output
            "--rm",  # Remove container after exit
        ]
    )

    # Add the image name
    cmd.append(image_name)

    # Build and add pipeline arguments (excluding data_path and output_dir)
    container_args = build_container_args(args)

    # Add container paths for data_path and output_dir
    container_args.extend([
        "--data_path", container_data_path,
        "--output_dir", container_output_path,
    ])

    cmd.extend(container_args)

    # Print the command for debugging
    print("Running Docker container with command:")
    print(" ".join([f'"{arg}"' if " " in arg else arg for arg in cmd]))
    print()

    try:
        # Run the container
        start_time = time.time()
        subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"Pipeline completed successfully in {elapsed_time:.1f} seconds.")
        print(f"Output saved to: {output_dir}.")

    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        return False
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        return False
    return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Pi3 Poisson pipeline Docker container",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with image directory
  python run_docker_pipeline.py --data_path ./images --output_dir ./results

  # Usage with video file and custom parameters
  python run_docker_pipeline.py --data_path ./video.mp4 --output_dir ./results \\
    --interval 5 --depth 12 --trim 8.0
        """,
    )

    # Add common pipeline arguments
    pipeline_config.add_common_args(parser)

    # Docker-specific arguments
    docker_group = parser.add_argument_group("Docker options")
    docker_group.add_argument("--container_name", default="pi3-poisson", help="Name of the container.")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input path
    data_path = pathlib.Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Input path does not exist: {data_path}")
        sys.exit(1)

    print("Pi3 Poisson Pipeline Docker Runner")

    # Run the Docker container
    print(f"Running Pipeline")
    success = run_docker_container(
        image_name=CONTAINER_NAME,
        args=args,
        container_name=args.container_name,
    )

    if success:
        print("Pipeline completed successfully.")
        sys.exit(0)
    else:
        print("Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
