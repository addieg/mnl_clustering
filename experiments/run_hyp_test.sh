#!/bin/bash
#SBATCH -a 1-33
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-00:00
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andreeag@mit.edu


  module load engaging/julia/0.6.1
  srun julia hypothesis_testing.jl $SLURM_ARRAY_TASK_ID 