import numpy as np
import matplotlib.pyplot as plt
import sys
from ase.io import read
from ase.calculators.emt import EMT
from mlvib.vibration_analysis import Vibration_analysis
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
    # Calculate using own code:
    vib = Vibration_analysis(molecule)
    vib.calculate_dynamic_matrix()
    w1 = vib.w
    # COMPARE THAT GARBAGE
    assert w1.shape == w2.shape
    wdiff = w2-w1


    np.set_printoptions(precision=4)

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


if __name__ == '__main__':
    from ase.io import read
    
    if sys.argv[1] == 'test2':
        import glob
        test_files = glob.glob('../EMT_test_molecules/*.traj')
        print(50*'#')
        print('|Molecule|')
        fail_count = 0
        passed = []; failed = []
        for file in test_files:
            state = compare_to_ASE(file=file, print_detailed=False)


            
            
        print('Total tests: {}'.format(len(test_files)))
        print('Number passed: {}'.format(len(test_files)-fail_count))
        print('Number failed {}'.format(fail_count))
        


