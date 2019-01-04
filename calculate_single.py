from ase.io import read, write
from ase.db import connect
import pickle
import argparse
import subprocess


subprocess.run('touch .start', shell=True)

# Parser input:
parser = argparse.ArgumentParser()
parser.add_argument('index', type=int)
parser.add_argument('traj_file', type=str)
args = parser.parse_args()

# Get atoms object
atoms = read(args.traj_file, index=args.index)

# GPAW calculator.
with open('calc.pckl', 'rb') as pickle_file:
    calc = pickle.load(pickle_file)
atoms.set_calculator(calc)
E = atoms.get_potential_energy()
F = atoms.get_forces()
D = atoms.get_dipole_moment()
write('atoms.traj', atoms)

subprocess.run('touch .finished', shell=True)
subprocess.run('rm .start', shell=True)








