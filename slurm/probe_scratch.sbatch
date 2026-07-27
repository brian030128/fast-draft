#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=probe_scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:02:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

echo "=== Worker scratch probe ==="
hostname
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
echo "TMPDIR=${TMPDIR:-unset}"
echo
echo "=== mount points ==="
df -h | grep -E "^/|tmpfs|nfs" | head -20
echo
echo "=== writable paths ==="
for p in /tmp /scratch /local /lscratch /work /mnt /data /staging; do
  if [ -d "$p" ] && [ -w "$p" ]; then
    df -h "$p" 2>&1 | tail -1
    ls -la "$p" 2>/dev/null | head -3
    echo "---"
  fi
done
echo
echo "=== try mkdir under common scratch ==="
for p in /tmp/$USER /scratch/$USER /local/$USER /lscratch/$USER /work/$USER; do
  if mkdir -p "$p" 2>/dev/null; then
    echo "OK: $p"
    df -h "$p" 2>&1 | tail -1
  fi
done
