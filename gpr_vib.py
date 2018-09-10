from ase import Atoms
from ase.vibrations import vibrations
from ase.calculators.emt import EMT
import ase.units as units

import sys
import numpy as np

class displacement:

    def __init__(self, index, direction, magnitude):
        self.index = index
        self.direction = direction
        self.magnitude = magnitude

    def __repr__(self):
        dirs = {0:'x', 1:'y', 2:'z'}
        direc = dirs[self.direction]
        return 'Displacement of atom {} in {}-direction by {}'.format(self.index, dirs[self.direction], self.magnitude)

class vibration_analysis():
    """
    Main class responsible for handling the vibrational analysis calculation.

    The inputs are:
    atoms: ASE atoms object
           Specifies geometry of the relaxed molecule
    delta: float [Å]
           Atomic displacements in Ångstrom

    """
    def __init__(self, atoms, delta=0.01):
        """
        Description of attributes:

        fast_calc: Calculator used to evaluate the elements of the dynamic matrix, i.e the GPR
        slow_calc: Calculator used to train the GPR e.g. GPAW DFT.
        """

        self.atoms = atoms
        self.delta = delta

        # Calculator stuff:
        self.fast_calc = EMT()
        self.slow_calc = None

        # Convenience:
        self.num_atoms = atoms.get_number_of_atoms() # Number of atoms
        self.Csize = self.num_atoms*3 # Size of Dynamic Matrix

    def calculate_displacements(self):
        """
        Calculate the displacements required to construct dynamic matrix.

        Does not include whether the displacement is positive or negative as both are done later.
        """
        self.displacements = []
        for index in range(self.num_atoms):
            for direction in range(3):
                self.displacements.append(displacement(index, direction, self.delta))

    def calculate_dynamic_matrix(self):
        """
        Constructs the dynamic matrix.

        To remember the index convention:
        fp[i, 0] = Force on the ith atom in the x-direction for a positive displacement. 
        """
        self.calculate_displacements()
        # This part calculates the dynamic matrix:
        C = np.zeros((self.Csize, self.Csize))
        for J, disp in enumerate(self.displacements):
            forces = []
            for sign in [1, -1]:
                atoms = self.atoms.copy()
                atoms.set_calculator(self.fast_calc)
                atoms[disp.index].position[disp.direction] += sign*disp.magnitude
                forces.append(atoms.get_forces())
            fp = forces[0]; fm = forces[1]
            C[:, J] = (fm-fp).ravel()/(2*disp.magnitude) # Eq 12 in Frederiksen et al.
        
        m = self.atoms.get_masses()**(-0.5)
        m = np.repeat(m, 3)
        w2, modes = np.linalg.eigh(m[:, None] * C * m)

        # Magic that turns calculated frequencies into eV (Stolen from ASE vibrations)
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        self.w = s * w2.astype(complex)**0.5


if __name__ == '__main__':
    from ase.io import read

    if sys.argv[1] == 'test1':
        atoms = read('EMT_test_molecules/O2.traj')
        A = vibration_analysis(atoms)
        A.calculate_dynamic_matrix()

    if sys.argv[1] == 'test2':
        import glob
        from vibrational.tests import test_functions
        test_files = glob.glob('EMT_test_molecules/*.traj')
        print(50*'#')
        print('|Molecule|')
        fail_count = 0
        passed = []; failed = []
        for file in test_files:
            state = test_functions.compare_to_ASE(file=file)


            
            
        print('Total tests: {}'.format(len(test_files)))
        print('Number passed: {}'.format(len(test_files)-fail_count))
        print('Number failed {}'.format(fail_count))
        
        




