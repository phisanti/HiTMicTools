import argparse
import sys
from pathlib import Path
from .main import main, main_new


def validate_file(file_path: str, extension: str = None) -> str:
    """Validate if file exists and has correct extension"""
    path = Path(file_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {file_path} does not exist")
    if extension and path.suffix != extension:
        raise argparse.ArgumentTypeError(f"File must have {extension} extension")
    return str(path)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="hitmictools",
        description="HiTMicTools: High-Throughput Microscopy Image Analysis Pipeline",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=lambda x: validate_file(x, ".yml"),
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "-w",
        "--worklist",
        type=lambda x: validate_file(x, ".txt") if x else None,
        default=None,
        help="Optional path to worklist text file",
    )

    return parser


def cli():
    parser = create_parser()
    args = parser.parse_args()

    try:
        main_new(args.config, args.worklist)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
