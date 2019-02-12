from ase.io import read
from mlvib.vibration_analysis import Vibration_analysis, vib_spectrum


pentane = read('pentane.mol')

vib = Vibration_analysis(pentane)
#vib.calculate_displacements()
#vib.write_displacements('pentane_disp.traj')

print(len(pentane))
vib.second_order_displacements()


#vib.calculate_dynamic_matrix()

#vib_spectrum(vib)

