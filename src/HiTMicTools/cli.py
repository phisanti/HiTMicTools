import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Any
from HiTMicTools.build_pipeline import build_and_run_pipeline


def validate_file(file_path: str, extension: str = None) -> str:
    """Validate if file exists and has correct extension"""
    path = Path(file_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {file_path} does not exist")
    if extension and path.suffix != extension:
        raise argparse.ArgumentTypeError(f"File must have {extension} extension")
    return str(path)


def add_split_files_command(subparsers):
    """Add the split-files command to the CLI"""
    parser = subparsers.add_parser(
        "split-files",
        help="Split files in a folder into multiple block files for parallel processing",
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        required=True,
        help="Path to the folder containing files to split",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        required=True,
        help="Number of blocks to split the files into",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./temp",
        help="Directory to save the block files (default: ./temp)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="Pattern to filter files (optional)",
    )
    parser.add_argument(
        "--file-extension",
        type=str,
        default=None,
        help="File extension to filter by (optional)",
    )
    parser.add_argument(
        "--return-full-path",
        action="store_true",
        default=True,
        help="If set, write full file paths to block files (default: True)",
    )
    parser.add_argument(
        "--no-return-full-path",
        dest="return_full_path",
        action="store_false",
        help="If set, write only file names to block files",
    )
    parser.set_defaults(return_full_path=True)
    parser.set_defaults(func=run_split_files)


def run_split_files(args):
    """Run the split-files command"""
    from HiTMicTools.batch_ops import split_files_into_blocks

    split_files_into_blocks(
        args.target_folder,
        args.n_blocks,
        args.output_dir,
        args.file_pattern,
        args.file_extension,
        args.return_full_path,
    )


def add_run_pipeline_command(subparsers):
    """Add the pipeline command to the CLI"""
    parser = subparsers.add_parser("run", help="Run the image analysis pipeline")
    parser.add_argument(
        "-c",
        "--config",
        type=lambda x: validate_file(x, ".yml"),
        required=True,
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "-w",
        "--worklist",
        type=str,
        default=None,
        help="Path to a worklist file containing file names to process",
    )
    parser.set_defaults(func=run_pipeline)


def run_pipeline(args):
    """Run the pipeline command"""
    build_and_run_pipeline(args.config, args.worklist)


def add_bundle_command(subparsers):
    """Add the bundle command to the CLI"""
    parser = subparsers.add_parser(
        "bundle",
        help="Create a model bundle (ZIP archive) from a configuration file"
    )
    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        required=True,
        help="Path to the YAML file describing models to bundle",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output path for the bundle (e.g., my_bundle.zip). Date will be auto-inserted.",
    )
    parser.add_argument(
        "--no-auto-date",
        action="store_true",
        help="Disable automatic date insertion in filename",
    )
    parser.set_defaults(func=run_bundle)


def run_bundle(args):
    """Run the bundle command"""
    from HiTMicTools.model_bundler import create_model_bundle

    auto_date = not args.no_auto_date
    output_path = create_model_bundle(
        args.input_config,
        args.output,
        auto_date=auto_date
    )
    print(f"Model bundle created successfully: {output_path}")


def add_generate_slurm_command(subparsers):
    """Add the generate-slurm command to the CLI"""
    parser = subparsers.add_parser(
        "generate-slurm", help="Generate a SLURM job submission template"
    )
    parser.add_argument(
        "--job-name", type=str, required=True, help="Name for the SLURM job"
    )
    parser.add_argument(
        "--email", type=str, default=None, help="Email for notifications"
    )
    parser.add_argument(
        "--time",
        type=str,
        default="06:00:00",
        help="Maximum allocated time (default: 06:00:00)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="rtx4090",
        help="SLURM partition (default: rtx4090)",
    )
    parser.add_argument(
        "--memory", type=str, default="25G", help="Memory per CPU (default: 25G)"
    )
    parser.add_argument(
        "--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)"
    )
    parser.add_argument(
        "--cpu-count", type=int, default=4, help="Number of CPUs per task (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./SLURM_jobs_report",
        help="Directory for SLURM output files (default: ./SLURM_jobs_report)",
    )
    parser.add_argument(
        "--config-file", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--file-blocks",
        action="store_true",
        help="Use file blocks for parallel processing",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=10,
        help="Number of blocks for parallel processing (default: 10)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="img_analysis",
        help="Conda environment name (default: img_analysis)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.getcwd(),
        help="Working directory (default: current working directory)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="slurm_job.sh",
        help="Output file for the SLURM script (default: slurm_job.sh)",
    )
    parser.set_defaults(func=run_generate_slurm)


def run_generate_slurm(args):
    """Run the generate-slurm command"""
    from HiTMicTools.batch_ops import generate_slurm_template

    template = generate_slurm_template(
        job_name=args.job_name,
        email=args.email,
        time=args.time,
        partition=args.partition,
        memory=args.memory,
        gpu_count=args.gpu_count,
        cpu_count=args.cpu_count,
        output_dir=args.output_dir,
        config_file=args.config_file,
        file_blocks=args.file_blocks,
        n_blocks=args.n_blocks,
        conda_env=args.conda_env,
        work_dir=args.work_dir,
    )

    with open(args.output_file, "w") as f:
        f.write(template)

    print(f"SLURM job script written to {args.output_file}")
    print(f"Submit with: sbatch {args.output_file}")


def hitmictools():
    """Main entry point for HiTMicTools CLI"""
    parser = argparse.ArgumentParser(
        description="HiTMicTools - High-Throughput Microscopy Analysis Tools"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add commands
    add_run_pipeline_command(subparsers)
    add_bundle_command(subparsers)
    add_split_files_command(subparsers)
    add_generate_slurm_command(subparsers)

    return parser
