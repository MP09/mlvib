from ase.io import read, write
import pickle
import argparse
import subprocess
import os



subprocess.run('touch .start', shell=True)

# Parser input:
parser = argparse.ArgumentParser()
parser.add_argument('start', type=int)
parser.add_argument('range', type=int, default=None)
parser.add_argument('traj_file', type=str)
parser.add_argument('jobID', type=int)
args = parser.parse_args()

with open('calc.pckl', 'rb') as pickle_file:
    calc = pickle.load(pickle_file)

top_folder = os.getcwd()
scratch_folder = '/scratch/{}/'.format(args.jobID)

start = args.start * args.range
stop = (args.start+1) * args.range

# Get atoms object
for j in range(start, stop):
    print('Starting calculation for job: {}'.format(j), flush=True)
    atoms = read(args.traj_file, index=j)
    cur_folder = top_folder + '/job_{}/'.format(j)


    if os.path.exists(cur_folder+'atoms.traj'):
        print('File already exists, continuing', flush=True)
        continue
    
    if not os.path.exists(cur_folder):
        try:
            os.mkdir(cur_folder)
        except:
            pass
        
    os.chdir(scratch_folder)
    atoms.set_calculator(calc)
    
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    D = atoms.get_dipole_moment()
    write(cur_folder+'atoms.traj', atoms)

    subprocess.run('cp *.out {}'.format(cur_folder+'.'), shell=True)
    subprocess.run('touch {}.finished'.format(cur_folder), shell=True)

    print('Finished job: {}'.format(j), flush=True)

    os.chdir(top_folder)


