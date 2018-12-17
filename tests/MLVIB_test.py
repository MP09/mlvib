from descriptors.behler_paranello import Behler_Paranello
from descriptors.averager import Averager, TypeAverager
from descriptors.local_to_global import Sorter

from mlvib.gaussian_process import GPCalculator
from mlvib.mlvib import MLVIB
from mlvib.vibration_analysis import Vibration_analysis, vib_spectrum
from ase.io import read
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import molecule
from timeit import default_timer as dt
import numpy as np

np.random.seed(1)

#atoms = read('pentane.mol')
#atoms = molecule('H2O')
positions = np.zeros((2, 3))
positions[0, 0] += 1
atoms = Atoms('CO', positions=positions)

#Setup descriptor:
eta = np.array([0.05])
rs = np.array([0])
xi = np.array([1])
lamb = np.array([1])
eta_ang = np.array([0.005])
rc = 2

parameters = {'eta':eta, 'rs':rs, 'xi':xi, 'lambda':lamb, 'eta_ang':eta_ang, 'rc':rc}
BP = Behler_Paranello(parameters)

avg = TypeAverager(BP)
sort = Sorter(BP, atomic=True)

# Stuff:
A = MLVIB(atoms, EMT(), sort)
A.calculate_displacements()
A.calculate_all_features()
#A.load_features()
A.initialize_GP()
A.calculate_spectrum()
A.main()


# for w in A.w:
#     print(w)



# GP = GPCalculator(A.GP, sort)

# # for jj in range(A.num_atoms*6):
# #     atoms = A.F_disp[jj]
# #     atoms.set_calculator(GP)

# #     F1 = atoms.get_forces().flatten()
# #     F2 = A.calculate_forces(E, jj)

# #     print((F1 == F2).all())



# vib = Vibration_analysis(atoms, GP)
# W = vib.calculate_dynamic_matrix()
# for w in W:
#     print(w)


#print(F)

