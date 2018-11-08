#!/bin/bash

#SBATCH --time=5:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per sub-job(i.e. tasks)
#SBATCH --mem-per-cpu=500M  # memory per CPU core
#SBATCH -J "16 turbs snopt relax for iea37 case 1"   # job name
#SBATCH --mail-user=jaredthomas68@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-199     # job array of size 100

echo ${SLURM_ARRAY_TASK_ID}

mpirun python case1_opt16.py ${SLURM_ARRAY_TASK_ID}
