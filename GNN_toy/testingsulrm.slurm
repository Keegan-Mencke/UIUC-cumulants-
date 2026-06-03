#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --partition=cpu        # or: general, batch, etc.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00        # HH:MM:SS
#SBATCH --mem=16G
#SBATCH --output=testing.out
#SBATCH --error=testing.err

# Load modules
module purge
module load python/3.10

# Optional: activate virtual environment
# source ~/venvs/myenv/bin/activate

# Run your program 
python testing_slurm.py