#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=probe_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:02:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

echo "=== hostname ==="
hostname
echo
echo "=== module system ==="
type module 2>&1 || echo "no module function"
ls /usr/share/lmod/lmod/init/bash 2>&1 || true
ls /etc/profile.d/ 2>&1 | head -20
echo
echo "=== gcc-toolset availability ==="
ls /opt/rh/ 2>&1 | head
ls /opt/ 2>&1 | head
echo
echo "=== /work/HPC_software/LMOD layout ==="
ls /work/HPC_software/LMOD 2>&1 | head
find /work/HPC_software/LMOD -maxdepth 4 -name "gcc-*" -o -name "gcc*" 2>/dev/null | head
echo
echo "=== nvcc + system gcc ==="
which nvcc; nvcc --version 2>&1 | tail -1
which gcc; gcc --version | head -1
echo
echo "=== try lmod via spack-like path ==="
[ -f /work/HPC_software/LMOD/lmod/lmod/init/bash ] && source /work/HPC_software/LMOD/lmod/lmod/init/bash && module avail 2>&1 | head -20 || echo "no lmod"
