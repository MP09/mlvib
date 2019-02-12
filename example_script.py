# Legit stuff:
from ase import Atoms
from ase.io import read
from gpaw import GPAW
import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from mlvib.igvpt import IGVPT
from mlvib.plotters import error_plot
from descriptors import BondsAngles
from ase.calculators.orca import ORCA

# Calculator:
calc = ORCA(label='orca',
               maxiter=2000,
               charge=0, mult=1,task='gradient',
               orcasimpleinput='LDA def2-TZVP ExtremeSCF',
               orcablocks='%scf maxiter 300 end \n%maxcore 1000 \n%pal nprocs 2 end')

# Read relaxed molecule:
atoms = read('relax/relax.traj')
atoms.set_calculator(calc)

# Setttings:
nmodes = 2
delta = 0.08 # Harmonics 
displacement = 0.04 # Anharmonics

# Passed to queueing script:
calc_settings = {}
calc_settings['task_per_node'] = 2
calc_settings['job_type'] = 'Orca'
calc_settings['max_core'] = 750
calc_settings['time'] = '48:00:00'


#energy_reference = [atoms.get_potential_energy() for atoms in read('QFF/testQFF_calced.traj', index=':')]

A = IGVPT(atoms, calc, nmodes=nmodes, delta=delta, calc_settings=calc_settings)
#A.calculate_anharmonics()

#Kernel:
kernel = RBF(length_scale=0.01, length_scale_bounds=(0.001, 2)) #+  WhiteKernel(noise_level=10**(-8))

#Descriptor:
parameters = {'use_angles':False}
descriptor = BondsAngles(parameters)


# Start stuff:
A.approximate_anharmonics_SK(descriptor, kernel, normalize=False, prior='mean_harmonic',
                             print_summary=True, n_restarts=1,
                             energy_reference=energy_reference)

# Final plot:
reference = np.array([2113.77])
error_plot(A.history, reference)


