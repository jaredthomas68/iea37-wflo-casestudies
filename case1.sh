#!/bin/bash

#SBATCH --time=15:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores per sub-job(i.e. tasks)
#SBATCH --mem-per-cpu=500G  # memory per CPU core
#SBATCH -J "38 turbs snopt relax"   # job name
#SBATCH --mail-user=jaredthomas68@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=16,36,64     # job array of size 100

echo ${SLURM_ARRAY_TASK_ID}

mpirun python case1_opt.py ${SLURM_ARRAY_TASK_ID}
