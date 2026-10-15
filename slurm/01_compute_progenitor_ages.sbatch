#!/bin/bash
#SBATCH --job-name=tprog_ages
#SBATCH --account=pi-rkessler
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log

# Step 01 on Midway: run from the repository root with  sbatch slurm/01_compute_progenitor_ages.sbatch
source /project2/rkessler/PRODUCTS/miniconda/etc/profile.d/conda.sh
conda activate /home/ymurakami/.conda/envs/py311
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python 01_compute_progenitor_ages.py --workers ${SLURM_CPUS_PER_TASK}
