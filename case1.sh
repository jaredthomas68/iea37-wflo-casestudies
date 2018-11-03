#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores per sub-job(i.e. tasks)
#SBATCH --mem-per-cpu=100G  # memory per CPU core
#SBATCH -J "16 turb case 1 iea37"   # job name
#SBATCH --mail-user=jaredthomas68@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=16,36,64 -N1 tmp    # job array of size 100

echo ${SLURM_ARRAY_TASK_ID}

python case1_opt.py ${SLURM_ARRAY_TASK_ID}
