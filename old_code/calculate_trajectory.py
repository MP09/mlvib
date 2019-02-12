import numpy as np
from math import ceil
import subprocess
from ase.io import Trajectory

s = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH --array={start}-{num_jobs}%{parallel_jobs}

echo "========= Job started  at `date` =========="

let SLURM_ARRAY_TASK_ID+={offset}

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
                'job_name':job_name, 'start':0,
                'parallel_jobs':min(num_jobs, max_parallel), 'time':time,
                'out_name':'slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out',
                'weird':'{SLURM_CPUS_PER_TASK:-1}', 'offset':0}

    if num_jobs < 2000:
        with open('job_file.sh', 'w') as job_file:
            print(s.format(**settings), file=job_file)
        subprocess.run('sbatch job_file.sh', shell=True)

    else:
        num_arrays = ceil(num_jobs / 2000)
        max_per_array = max_parallel // num_arrays

        settings['num_jobs'] = 1999
        settings['parallel_jobs'] = max_per_array
        
        for i in range(num_arrays):

            settings['offset'] = i*2000

            with open('job_file{}.sh'.format(i), 'w') as job_file:
                print(s.format(**settings), file=job_file)
            
            subprocess.run('sbatch job_file{}.sh'.format(i), shell=True)


            
        
        

    

    
    

    

    

    

