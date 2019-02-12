
from math import ceil
import subprocess
import os

s = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --mem={memory}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={task_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --time={time}
#SBATCH --array=0-{num_array}%{num_parallel}
#SBATCH --output={out_name}

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"

source /home/machri/envs/python3.6.3_new/bin/activate

export OMP_NUM_THREADS=${weird}

{run_command} /home/machri/PythonPackages/mlvib/calculate_range.py $SLURM_ARRAY_TASK_ID {jobs_per_array} {traj_file} $SLURM_JOB_ID

echo "========= Job finished at `date` =========="
"""

def calculate_trajectory(set_dict):
    """
    Queues calculations for all structures in a trajectory file.
    
    Expects that there is a 'calc.pckl' file in the trajectory this is called from.
    """
    
    if not os.path.exists('out'):
        os.mkdir('out')

    # Setup basic settings dictionary:
    settings = {}

    # These are required:
    settings['traj_file'] = None #traj_file
    settings['num_jobs'] = None  #num_jobs
    settings['job_name'] = None  # job_name

    # These are not required and set to default:
    settings['memory'] = '4G'
    settings['num_nodes'] = 1
    settings['max_core'] = 500
    settings['task_per_node'] = 1
    settings['time'] = '01:00:00'
    settings['jobs_per_array'] = 1
    settings['job_type'] = 'GPAW'
    settings['partition'] = 'q20,q16,q12'
    settings['weird'] = '{SLURM_CPUS_PER_TASK:-1}'
    settings['out_name'] = 'out/slurm-%A_%a.out'

    # Set dict:
    for key in set_dict.keys():
        if key in settings.keys():
            settings[key] = set_dict[key]
        else:
            print('Bad key: {}'.format(key))
    
    # Determine how many jobs each array should run:
    if settings['max_core'] < settings['num_jobs'] * settings['task_per_node']:
        settings['num_array'] = ceil(settings['max_core'] / settings['task_per_node'])
        settings['jobs_per_array'] = ceil(settings['num_jobs'] / settings['num_array'])
    else:
        settings['num_array'] = settings['num_jobs']
    settings['num_parallel'] = settings['num_array']
    settings['num_array'] -= 1


    # Determine the job_type:
    if settings['job_type'] == 'GPAW':
        settings['run_command'] = 'srun -n {} gpaw-python'.format(settings['task_per_node'])
    else:
        settings['run_command'] = 'python'

    with open('job_file.sh', 'w') as f:
        print(s.format(**settings), file=f)
        
    subprocess.run('sbatch job_file.sh', shell=True)


if __name__ == '__main__':

    set_dict = {}
    set_dict['max_core'] = 50
    set_dict['task_per_node'] = 3
    set_dict['job_type'] = 'ORCA'

    set_dict['traj_file'] = 'traj_file.traj'
    set_dict['num_jobs'] = 16
    set_dict['job_name'] = 'bad_job'

    calculate_trajectory(set_dict)
    

