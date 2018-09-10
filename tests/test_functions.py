import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.calculators.emt import EMT
from ML_vib import vibration_analysis
from ase.vibrations import Vibrations

def compare_to_ASE(molecule=None, file=None, threshold=1, print_detailed=False):
    if molecule == None:
        molecule = read(file)

    # Calculate using ASE code:
    molecule.set_calculator(EMT())
    vib = Vibrations(molecule)
    try:
        vib.clean()
    except:
        pass
    vib.run()
    w2 = vib.get_energies()
    C2 = vib.Q
    # Calculate using own code:
    vib = vibration_analysis(molecule)
    C1 = vib.calculate_dynamic_matrix()
    #C3 = vib.calculate_dynamic_matrix()
    
    np.set_printoptions(precision=4)
    
   

    
    

    w1 = vib.w
    # COMPARE THAT GARBAGE
    assert w1.shape == w2.shape
    wdiff = w2-w1




    P = abs((w2-w1)/w2*100)
    idx1 = (w1.real > 10**(-4))*(w2.real > 10**(-4))
    state = (P.real[idx1] < threshold).all()

    if print_detailed:
        print('Molecule: {}'.format(molecule.get_chemical_formula()))
        print('Number of atoms in molecule: {}'.format(molecule.get_number_of_atoms()))
        for i in range(len(w1)):
            if P[i].real < threshold:
                good = 'Pass '
            else:
                good = 'Fail '


            if idx1[i]:
                print(good+'C{}: {:7.6f}, {:7.6f}, {:4.3f}'.format(i, w1[i].real, w2[i].real, P[i].real))
            else:
                print(good+'N{}: {:7.6f}, {:7.6f}, {:4.3f}'.format(i, w1[i].real, w2[i].real, P[i].real))
        #print(fp1-fp2)

    #vib.clean()

    name = molecule.get_chemical_formula()
    
    return name, state




