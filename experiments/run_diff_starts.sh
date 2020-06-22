#!/bin/bash
#SBATCH -a 1-11
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andreeag@mit.edu


  module load engaging/julia/0.6.1
  srun julia try_diff_starts.jl $SLURM_ARRAY_TASK_ID 