import os
import math
from pathlib import Path
from typing import List, Optional


def split_files_into_blocks(
    target_folder: str,
    n_blocks: int,
    output_dir: str = "./temp",
    file_pattern: Optional[str] = None,
    file_extension: Optional[str] = None,
) -> List[str]:
    """
    Split files in a target folder into multiple block files for parallel processing.

    Args:
        target_folder (str): Path to the folder containing files to split
        n_blocks (int): Number of blocks to split the files into
        output_dir (str, optional): Directory to save the block files. Defaults to "./temp".
        file_pattern (str, optional): Pattern to filter files. Defaults to None.
        file_extension (str, optional): File extension to filter by. Defaults to None.

    Returns:
        List[str]: List of paths to the created block files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of files in the target folder
    files = os.listdir(target_folder)

    # Apply filters if specified
    if file_extension:
        files = [f for f in files if f.endswith(file_extension)]
    if file_pattern:
        files = [f for f in files if file_pattern in f]

    if not files:
        print(f"No matching files found in {target_folder}")
        return []

    # Calculate files per block
    files_per_block = math.ceil(len(files) / n_blocks)

    # Create block files
    block_files = []
    for i in range(min(n_blocks, len(files))):
        start_idx = i * files_per_block
        end_idx = min((i + 1) * files_per_block, len(files))

        if start_idx >= len(files):
            break

        block_path = os.path.join(output_dir, f"file_block_{i}.txt")
        with open(block_path, "w") as f:
            for file_name in files[start_idx:end_idx]:
                f.write(f"{file_name}\n")

        block_files.append(block_path)

    print(f"Created {len(block_files)} block files in {output_dir}")
    return block_files


def generate_slurm_template(
    job_name: str,
    email: str = None,
    time: str = "06:00:00",
    partition: str = "rtx4090",
    memory: str = "25G",
    gpu_count: int = 1,
    cpu_count: int = 4,
    output_dir: str = "./SLURM_jobs_report",
    config_file: str = None,
    file_blocks: bool = False,
    n_blocks: int = 10,
    conda_env: str = "img_analysis",
    work_dir: str = os.getcwd(),
) -> str:
    """
    Generate a SLURM job submission template for HiTMicTools pipeline.

    Args:
        job_name (str): Name for the SLURM job
        email (str, optional): Email for notifications. Defaults to None.
        time (str, optional): Maximum allocated time. Defaults to "06:00:00".
        partition (str, optional): SLURM partition. Defaults to "rtx4090".
        memory (str, optional): Memory per CPU. Defaults to "25G".
        gpu_count (int, optional): Number of GPUs. Defaults to 1.
        cpu_count (int, optional): Number of CPUs per task. Defaults to 4.
        output_dir (str, optional): Directory for SLURM output files. Defaults to "./SLURM_jobs_report".
        config_file (str, optional): Path to config file. Defaults to None.
        file_blocks (bool, optional): Whether to use file blocks for parallel processing. Defaults to False.
        n_blocks (int, optional): Number of blocks for parallel processing. Defaults to 10.
        conda_env (str, optional): Conda environment name. Defaults to "img_analysis".
        work_dir (str, optional): Working directory. Defaults to ".".

    Returns:
        str: SLURM job submission script content
    """

    # Create base template
    template = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name={job_name}",
    ]

    # Add email if provided
    if email:
        template.extend(
            [f"#SBATCH --mail-user={email}", "#SBATCH --mail-type=END,FAIL"]
        )

    # Add resource specifications
    template.extend(
        [
            f"#SBATCH --time={time}",
            "#SBATCH --qos=gpu6hours",
            f"#SBATCH --mem-per-cpu={memory}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --gres=gpu:{gpu_count}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={cpu_count}",
            f"#SBATCH --output={output_dir}/{job_name}/%j_HiTMicTools.out",
            f"#SBATCH --error={output_dir}/{job_name}/%j_HiTMicTools.err",
        ]
    )

    # Add array job if using file blocks
    if file_blocks:
        template.append(f"#SBATCH --array=0-{n_blocks - 1}")

    # Add modules and environment setup
    template.extend(
        [
            "",
            "# 1. Load modules",
            "module load Python",
            "module load CUDA",
            "module load jobstats",
            f"mkdir -p {output_dir}/{job_name}",
            "",
            "# Check GPU/CUDA",
            "if command -v nvidia-smi &> /dev/null",
            "then",
            '  echo  "GPU information:"',
            "    nvidia-smi",
            "else",
            '    echo "No GPU available"',
            "fi",
            "",
            "# Check CPU",
            'echo "CPU information:"',
            "lscpu | egrep 'Model name|Socket|Thread|NUMA|CPU\\(s\\)'",
            "",
            "# Set conda environment",
            "source ~/.bashrc",
            "conda activate",
            "conda init",
            f"conda activate {conda_env}",
            "",
            f"# Change to working directory",
            f"cd '{work_dir}'",
            "",
        ]
    )

    # Add execution commands
    if file_blocks:
        temp_dir = f"./temp_{job_name}"
        temp_dir2 = f"./temp2_{job_name}"

        template.extend(
            [
                "# Run analysis with file blocks",
                "BLOCK_NUM=$((SLURM_ARRAY_TASK_ID))",
                'echo "Analysis starts now"',
                'echo "Processing file_block_${BLOCK_NUM}"',
                "",
                f"# Generate file blocks if they don't exist",
                f'if [ ! -d "{temp_dir}" ]; then',
                f"    hitmictools split-files --target-folder . --n-blocks {n_blocks} --output-dir {temp_dir}",
                "fi",
                "",
            ]
        )

        if config_file:
            template.extend(
                [
                    'echo "Config file contents:"',
                    f'cat "{config_file}"',
                    "",
                    'echo "Executing command:"',
                    f'echo "hitmictools run --config {config_file} --worklist {temp_dir}/file_block_${{BLOCK_NUM}}.txt"',
                    'echo ""',
                    f"hitmictools run --config {config_file} --worklist {temp_dir}/file_block_${{BLOCK_NUM}}.txt",
                    "",
                    "# Process any failed files",
                    f"mkdir -p {temp_dir2}",
                    "OUTPUT_DIR=$(grep -A1 'output_dir:' \"$CONFIG_FILE\" | tail -n1 | sed 's/^[ \\t]*//')",
                    'PROCESSED_FILES=$(find "$OUTPUT_DIR" -name "*_fl.csv" | sed \'s/.*\\///; s/_fl\\.csv$//\')',
                    "",
                    f'BLOCK_FILE="{temp_dir}/file_block_${{BLOCK_NUM}}.txt"',
                    f'OUTPUT_FILE="{temp_dir2}/file_block_${{BLOCK_NUM}}.txt"',
                    'grep -vFf <(printf "%s\\n" "${PROCESSED_FILES[@]}") "$BLOCK_FILE" > "$OUTPUT_FILE"',
                    'echo "$(wc -l < "$OUTPUT_FILE") files remaining to process"',
                    "",
                    'if [ -s "$OUTPUT_FILE" ]; then',
                    '    echo "Retrying failed files"',
                    f'    hitmictools run --config {config_file} --worklist "$OUTPUT_FILE"',
                    "fi",
                ]
            )
        else:
            template.append(
                "# Please specify a config file with --config-file parameter"
            )
    else:
        # Single job mode
        if config_file:
            template.extend(
                [
                    'echo "Config file contents:"',
                    f'cat "{config_file}"',
                    "",
                    'echo "Executing command:"',
                    f'echo "hitmictools run --config {config_file}"',
                    'echo ""',
                    f"hitmictools run --config {config_file}",
                ]
            )
        else:
            template.append(
                "# Please specify a config file with --config-file parameter"
            )

    return "\n".join(template)
