import numpy as np
from ase.io import Trajectory

s = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH --array=0-{num_jobs}%{parallel_jobs}

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"

source /home/machri/envs/python3.6.3_new/bin/activate

export OMP_NUM_THREADS=${weird}

mkdir job_$SLURM_ARRAY_TASK_ID
cd job_$SLURM_ARRAY_TASK_ID
cp ../calc.pckl .
cp /home/machri/PythonPackages/mlvib/calculate_single.py .

python calculate_single.py $SLURM_ARRAY_TASK_ID ../{traj_file}

cd ..
mv {out_name} out/.


echo "========= Job finished at `date` =========="
"""

def calculate_trajectory(traj_file, num_jobs, job_name, time='01:00:00', max_parallel=500):

    settings = {'traj_file':traj_file, 'partition':'q20,q16,q12', 'num_jobs':num_jobs-1,
                'job_name':job_name,'weird':'{SLURM_CPUS_PER_TASK:-1}',
                'parallel_jobs':min(num_jobs, max_parallel), 'time':time,
                'out_name':'slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out'}

    with open('job_file.sh', 'w') as job_file:
        print(s.format(**settings), file=job_file)


    

    
    

    

    

    

