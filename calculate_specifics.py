import numpy as np
from ase.io import Trajectory

s = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --array=1-{num_jobs}%{parallel_jobs}

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"

source /home/machri/envs/python3.6.3_new/bin/activate

export OMP_NUM_THREADS=${weird}

folder_ID=$(awk -v idx=$SLURM_ARRAY_TASK_ID 'NR == idx {awk}' runfile)

mkdir job_$folder_ID
cd job_$folder_ID
cp ../calc.pckl .
cp /home/machri/PythonPackages/mlvib/calculate_single.py .
cp ../runfile .

echo `awk "NR == $SLURM_ARRAY_TASK_ID" runfile`

python calculate_single.py `awk "NR == $SLURM_ARRAY_TASK_ID" runfile`
cd ..

echo "========= Job finished at `date` =========="
"""

def calculate_specifics(idxs, traj_file, num_jobs, job_name):

    settings = {'traj_file':traj_file, 'num_jobs':num_jobs,
                'partition':'q20,q16,q12', 'job_name':job_name,
                'weird':'{SLURM_CPUS_PER_TASK:-1}', 'parallel_jobs':num_jobs,
                'out_name':'slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out',
                'awk':'{print $1}'}

    with open('job_file.sh', 'w') as job_file:
        print(s.format(**settings), file=job_file)

    with open('runfile', 'w') as run_file:
        for idx in idxs:
            print('{} ../{}'.format(idx, traj_file), file=run_file)
            
