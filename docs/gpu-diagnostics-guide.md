# GPU Diagnostics Guide

## Overview

The `hitmictools gpu-check` command provides comprehensive GPU diagnostics to help detect and troubleshoot GPU-related issues in HiTMicTools pipelines. This is particularly useful when running on HPC clusters where GPU availability and configuration can be complex.

## Quick Start

### Basic Usage

```bash
# Run basic GPU diagnostics
hitmictools gpu-check

# Save detailed output to file
hitmictools gpu-check --output gpu_report.txt

# Show verbose output (includes full environment variables)
hitmictools gpu-check --verbose
```

### On HPC/SLURM Cluster

For sciCORE or other SLURM-based HPC systems:

```bash
# Use the provided SLURM script
sbatch scripts/gpu_diagnostic_slurm.sh

# Or run interactively in a GPU job
srun --partition=a100 --gres=gpu:1 --pty bash
conda activate img_analysis
hitmictools gpu-check --verbose
```

## What the Diagnostic Checks

The GPU diagnostic tool performs comprehensive checks across multiple layers:

### 1. System Information
- Operating system and kernel version
- Hostname (useful for identifying problematic nodes)
- Python version and installation path
- HPC/SLURM environment detection

### 2. SLURM Environment (if applicable)
- SLURM Job ID
- GPU allocation (`SLURM_GPUS_ON_NODE`, `SLURM_JOB_GPUS`)
- Detects if GPUs were requested but not allocated

### 3. Environment Modules
- Lists loaded modules (CUDA, Python, etc.)
- Helps verify correct module versions are loaded

### 4. Environment Variables
- `CUDA_HOME`, `CUDA_PATH`: CUDA installation paths
- `CUDA_VISIBLE_DEVICES`: Which GPUs are visible
- `LD_LIBRARY_PATH`: Library search paths
- `TMPDIR`: Temporary directory (important for resource management)

### 5. NVIDIA Driver (nvidia-smi)
- Driver version
- CUDA version supported by driver
- GPU model, memory, temperature, utilization
- Detects unresponsive or locked GPUs

### 6. PyTorch CUDA
- PyTorch version and CUDA support
- CUDA version built into PyTorch
- cuDNN availability
- GPU count and properties (compute capability, memory)
- **GPU compute test**: Actually creates tensors and performs matrix multiplication

### 7. Diagnostic Summary
- Clear pass/fail status for each component
- Prioritized list of issues found
- Specific troubleshooting steps for each issue

## Common Issues and Solutions

### Issue 1: "nvidia-smi not available"

**Symptoms:**
```
NVIDIA Driver (nvidia-smi):
  Status: Not available
  Error: nvidia-smi failed: command not found
```

**Solutions:**
1. Verify you're on a GPU node: `hostname`
2. Check GPU allocation in SLURM job: `echo $SLURM_GPUS_ON_NODE`
3. If on login node, submit a GPU job first
4. Check if NVIDIA driver is installed: `lspci | grep -i nvidia`

### Issue 2: "PyTorch CUDA not available"

**Symptoms:**
```
PyTorch CUDA:
  CUDA Available: False
  Error: CUDA is not available to PyTorch
```

**Solutions:**

#### On HPC Systems:
1. **Load CUDA module**: `module load CUDA`
2. **Check module compatibility**: Ensure CUDA module version matches PyTorch
   ```bash
   module avail CUDA  # See available versions
   python -c "import torch; print(torch.version.cuda)"
   ```
3. **Check for conflicting packages**:
   ```bash
   conda list | grep cuda
   # Remove conda-installed CUDA if using system modules
   conda remove cudatoolkit
   ```
4. **Verify GPU allocation in SLURM**:
   ```bash
   # Check your SLURM script has:
   #SBATCH --gres=gpu:1
   ```

#### On Local Systems:
1. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue 3: "GPU tensor operations failed"

**Symptoms:**
```
GPU Compute Test (tensor operations):
  Status: FAILED
  Error: CUDA out of memory / GPU not responding
```

**Solutions:**
1. **GPU may be locked**: Check with `nvidia-smi`
2. **Another process using GPU**: Run `nvidia-smi pmon` to see processes
3. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. **GPU hardware issue**: Try a different GPU node or contact HPC support

### Issue 4: "Running in SLURM but no GPUs allocated"

**Symptoms:**
```
! WARNING: Running in SLURM but no GPUs allocated to job
```

**Solutions:**
1. Add `--gres=gpu:1` to your `sbatch` or `srun` command
2. Check your SLURM script has the `#SBATCH --gres=gpu:N` line
3. Verify partition supports GPUs: `sinfo -o "%P %G"` (look for `gpu:` in output)

## Integration with Pipeline Runs

### Before Running HiTMicTools Pipeline

Always verify GPU availability before starting large pipeline runs:

```bash
# 1. Submit a quick diagnostic job
sbatch scripts/gpu_diagnostic_slurm.sh

# 2. Check the output
cat gpu_diagnostics_detailed_*.txt

# 3. If all checks pass, submit your pipeline job
sbatch your_pipeline_job.sh
```

### Troubleshooting Failed Pipeline Runs

If a pipeline fails with GPU errors:

```bash
# 1. SSH to the node where the job ran (from SLURM output)
ssh <node_name>

# 2. Run diagnostics on that specific node
srun --partition=a100 --nodelist=<node_name> --gres=gpu:1 \
     hitmictools gpu-check --verbose --output gpu_issue_report.txt

# 3. Share the report with HPC support if needed
```

## Command Reference

### Syntax
```
hitmictools gpu-check [OPTIONS]
```

### Options
- `-o, --output FILE`: Save diagnostics to file (appends to existing file)
- `-v, --verbose`: Show detailed output including full environment paths
- `-h, --help`: Show help message

### Exit Codes
- `0`: All GPU checks passed
- `1`: One or more GPU checks failed

### Example Workflows

**Quick check during interactive session:**
```bash
srun --partition=rtx4090 --gres=gpu:1 --pty bash
conda activate img_analysis
hitmictools gpu-check
```

**Comprehensive diagnostic with report:**
```bash
hitmictools gpu-check --verbose --output ~/gpu_reports/$(date +%Y%m%d)_diagnostic.txt
```

**Check multiple nodes:**
```bash
# Create a script to check all GPU nodes
for node in gpu-node-{01..10}; do
    srun --nodelist=$node --gres=gpu:1 \
         hitmictools gpu-check --output gpu_node_survey.txt
done
```

## Tips for HPC Users

1. **Save diagnostics to persistent storage**: Use `--output` to save reports to your home directory, not `$TMPDIR`

2. **Run diagnostics after module changes**: If you load different CUDA or Python modules, re-run diagnostics

3. **Check before large batch jobs**: A 5-minute diagnostic can save hours of failed pipeline runs

4. **Share reports with collaborators**: When reporting GPU issues, attach the diagnostic output

5. **Compare working vs. broken setups**: Run diagnostics on a known-working node and a problematic node, then compare

## Getting Help

If GPU issues persist after following troubleshooting steps:

1. Run `hitmictools gpu-check --verbose --output gpu_issue.txt`
2. Include the output file when contacting:
   - HPC support (for driver/allocation issues)
   - HiTMicTools maintainers (for PyTorch/pipeline issues)
3. Mention:
   - Cluster name and partition
   - SLURM job ID (if applicable)
   - Node hostname
   - Loaded modules (`module list`)
