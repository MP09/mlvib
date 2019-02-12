from descriptors.behler_paranello import Behler_Paranello
from descriptors.averager import Averager, TypeAverager

from mlvib.mlvib import MLVIB
from mlvib.vibration_analysis import Vibration_analysis, vib_spectrum
from ase.io import read
from ase.calculators.emt import EMT
from ase.build import molecule
from timeit import default_timer as dt
import numpy as np

#atoms = read('pentane.mol')
atoms = molecule('H2O')



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

# Stuff:
A = MLVIB(atoms, EMT(), avg)
A.initialize_GP()

t0 = dt()
A.calculate_spectrum()
print(dt()-t0)

#import matplotlib.pyplot as plt 
#vib = Vibration_analysis(atoms, calc=EMT())
#vib.calculate_dynamic_matrix()
#for w in vib.w:
#    print(w.real)



# fig, ax = plt.subplots()
# vib_spectrum(A.vib, fig, ax)
# vib_spectrum(vib, fig, ax)
# plt.show()


