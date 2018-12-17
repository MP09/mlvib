from ase import Atoms
from ase.vibrations import vibrations
from ase.calculators.emt import EMT
from ase.io import Trajectory
import ase.units as units

import sys
import numpy as np
import matplotlib.pyplot as plt

class displacement:

    def __init__(self, index, direction, magnitude):
        self.index = index
        self.direction = direction
        self.magnitude = magnitude

    def __repr__(self):
        dirs = {0:'x', 1:'y', 2:'z'}
        direc = dirs[self.direction]
        return 'Displacement of atom {} in {}-direction by {}'.format(self.index, dirs[self.direction], self.magnitude)

class Vibration_analysis():
    """
    Main class responsible for handling the vibrational analysis calculation.

    The inputs are:
    atoms: ASE atoms object
           Specifies geometry of the relaxed molecule
    delta: float [Å]
           Atomic displacements in Ångstrom

    """
    def __init__(self, atoms, calc=None, delta=0.01, delta2=0.001):
        """
        Description of attributes:
nn
        fast_calc: Calculator used to evaluate the elements of the dynamic matrix, i.e the GPR
        slow_calc: Calculator used to train the GPR e.g. GPAW DFT.
        """

        self.atoms = atoms
        self.delta = delta
        self.delta2 = delta2

        # Calculator stuff:
        self.calc = calc

        # Convenience:
        self.num_atoms = atoms.get_number_of_atoms() # Number of atoms
        self.Csize = self.num_atoms*3 # Size of Dynamic Matrix

    def set_calculator(self, calc):
        self.calc = calc
        
    def calculate_displacements(self):
        """
        Calculate the displacements required to construct dynamic matrix.

        Does not include whether the displacement is positive or negative as both are done later.
        """
        self.displacements = []
        for index in range(self.num_atoms):
            for direction in range(3):
                self.displacements.append(displacement(index, direction, self.delta))

    def get_displacements(self):
        if not hasattr(self, 'displacements'):
            self.calculate_displacements()

        displacements = []
        for disp in self.displacements:
            for sign in [-1, 1]:
                temp = self.atoms.copy()
                temp.positions[disp.index, disp.direction] += sign*disp.magnitude
                displacements.append(temp)
        return displacements

    def second_order_displacements(self):
        if not hasattr(self, 'displacements'):
            self.calculate_displacements()
    
        disps = []
        for disp in self.displacements:
            for sign in [-1, 1]:
                for a in range(len(self.atoms)):
                    for i in range(3):
                        for sign2 in [-1, 1]:
                            temp = self.atoms.copy()
                            temp.positions[disp.index, disp.direction] += sign*disp.magnitude
                            temp.positions[a, i] += sign*self.delta2
                            disps.append(temp)
        print(len(disps))
        

    def write_displacements(self, traj_name):
        """
        Create trajectory file containing all the displacements required. 
        """
        trajectory = Trajectory(traj_name, mode='w')
        for disp in self.displacements:
            for sign in [-1, 1]:
                temp = self.atoms.copy()
                temp.positions[disp.index, disp.direction] += sign*disp.magnitude
                trajectory.write(temp)
                
    def calculate_dynamic_matrix(self, return_mode=False):
        """
        Constructs the dynamic matrix.

        To remember the index convention:
        fp[i, 0] = Force on the ith atom in the x-direction for a positive displacement. 
        """
        assert self.calc is not None
        
        self.calculate_displacements()
        # This part calculates the dynamic matrix:
        C = np.zeros((self.Csize, self.Csize))
        for J, disp in enumerate(self.displacements):
            forces = []
            for sign in [1, -1]:
                atoms = self.atoms.copy()
                atoms.set_calculator(self.calc)
                atoms[disp.index].position[disp.direction] += sign*disp.magnitude
                forces.append(atoms.get_forces())
            fp = forces[0]; fm = forces[1]
            C[:, J] = (fm-fp).ravel()/(2*disp.magnitude) # Eq 12 in Frederiksen et al.

        self.C = C
            
        m = self.atoms.get_masses()**(-0.5)
        m = np.repeat(m, 3)
        w2, modes = np.linalg.eigh(m[:, None] * C * m)
        modes = modes.T

        # Magic that turns calculated frequencies into eV (Stolen from ASE vibrations)
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        self.w = s * w2.astype(complex)**0.5
        if not return_mode:
            return self.w
        else:
            modes_out = np.zeros((len(modes), len(self.atoms), 3))
            for i in range(len(modes)):
                modes_out[i] = (modes[i] * m).reshape(-1, 3)
            
            return self.w, modes_out

    def get_hessian(self):
        if hasattr(self, 'C'):
            return self.C
        else:
            self.calculate_dynamic_matrix()
            return self.C


def vib_spectrum(vib, fig=None, ax=None):

    sigma = 0.0025
    gauss = lambda x, x1: np.exp(-(x-x1)**2/sigma**2)

    freq = vib.w[vib.w > 10**(-10)].real
    
    x = np.linspace(0, max(freq)*1.1, 500)
    spec = np.zeros_like(x)
    
    for w in freq:

        spec += gauss(x, w)

    if fig == None:
        fig, ax = plt.subplots()
    
    ax.plot(x, spec)

    plt.xlim([min(x), max(x)])
    plt.tight_layout()
    

        
    
        
        
if __name__ == '__main__':
    from ase.io import read

    if sys.argv[1] == 'test1':
        atoms = read('EMT_test_molecules/O2.traj')
        A = vibration_analysis(atoms)
        A.calculate_dynamic_matrix()

    if sys.argv[1] == 'test2':
        import glob
        from mlvib.tests import test_functions
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
        
        




