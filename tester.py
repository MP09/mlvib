from descriptors.behler_paranello import Behler_Paranello
from descriptors.averager import Averager, TypeAverager
from descriptors.local_to_global import Sorter

from mlvib.gaussian_process import GPCalculator
from mlvib.mlvib import MLVIB
from mlvib.vibration_analysis import Vibration_analysis

from ase.io import read, write
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.build import molecule

import numpy as np
import os

def test(atoms, name):
    folder = name + '/'
    if not os.path.exists(folder):
        os.mkdir(folder)

    
    atoms.set_calculator(EMT())
    dyn = BFGS(atoms)
    dyn.run(fmax=0.005)
    write(folder+'relax.traj', atoms)

    # Full calculation:
    vib = Vibration_analysis(atoms, EMT())
    w = vib.calculate_dynamic_matrix()
    np.save(folder+'w_EMT.npy', w)
    
    # Setup descriptor:
    eta = np.array([0.05, 1, 4, 8])
    rs = np.array([0, 0, 0, 0])
    xi = np.array([1, 2, 1, 2]).astype(np.float)
    lamb = np.array([1, -1, 1, -1])
    eta_ang = np.array([0.005, 0.005, 0.005, 0.005])
    rc = 2

    parameters = {'eta':eta, 'rs':rs, 'xi':xi, 'lambda':lamb, 'eta_ang':eta_ang, 'rc':rc}
    BP = Behler_Paranello(parameters)
    sort = Sorter(BP, atomic=True)
    
    # Stuff:    
    A = MLVIB(atoms, EMT(), sort)
    A.folder = folder
    A.calculate_displacements()
    A.calculate_all_features()
    A.initialize_GP()
    A.main()

    


