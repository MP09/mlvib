import numpy as np
import ase.units as units
from timeit import default_timer as dt
from mlvib.gaussian_process import Gaussian_process, GPCalculator
from mlvib.vibration_analysis import Vibration_analysis
from mlvib.kernels import RBF

class MLVIB:

    def __init__(self, atoms, calc, descriptor):
        """
        Main class for calculating vibrational frequencies using Gaussian Process Regression. 

        Inputs:
        -- atoms: ASE atoms object.
           Relaxed atoms object of the molecule. 
        -- calc: ASE calc object.
           Calculator to fit to. 
        """

        # Atoms object:
        self.atoms = atoms
        self.num_atoms = len(atoms)

        # 'Slow calculator' i.e. DFT:
        self.calc = calc

        # Descriptor object:
        self.descriptor = descriptor

        # Displacement settings:
        self.deltaF = 0.01 
        self.deltaS = 0.001

        # Common numbers:
        self.num_first = 6*self.num_atoms**2
        self.num_second = 36*self.num_atoms**2

        # Folder:
        self.folder = 'data/'

        print('Number of displacements: {}'.format(self.num_second))
        
    
    def calculate_displacements(self):
        """
        Calculate first and second order displacements.

        First order are saved in a list of length 6N
        
        1st entry: atom i in positive x-direction.
        2nd entry: atom i in negative x-direction.

        Second order in a list of length 36N^2
        """

        self.F_disp = []
        self.S_disp = []

        # Calculate first order displacement:
        for a in range(self.num_atoms): # Atom
            for i in [0, 1, 2]:         # Axis
                for sign in [-1, 1]:    # Direction
                    atoms = self.atoms.copy()
                    atoms.positions[a, i] += sign*self.deltaF
                    self.F_disp.append(atoms)

        # Calculate second order displacements:
        for Fatoms in self.F_disp:
            for sign in [-1, 1]:
                for a in range(self.num_atoms):
                    for i in [0, 1, 2]:
                        atoms = Fatoms.copy()
                        atoms.positions[a, i] += sign*self.deltaS
                        self.S_disp.append(atoms)
                        
    def calculate_all_features(self, save=True):
        """
        Calculate feature vectors for all second order displacements.
        """ 
        print('Calculating features..')
        t0 = dt()
        
        # Calculate first feature outside loop to get dimensions:
        F = self.descriptor(self.S_disp[0])

        # Make array:
        self.features = np.zeros((len(self.S_disp), F.size))
        self.features[0, :] = F

        for a, atoms in enumerate(self.S_disp[1::]):
            self.features[a+1, :] = self.descriptor(atoms)

        if save:
            np.save(self.folder+'feature_dump.npy', self.features)

        print('Feature calculation took: {} s'.format(dt()-t0))
        
    def load_features(self, file_name='feature_dump.npy', folder=''):
        self.features = np.load(folder+file_name)
        assert self.features.shape[0] == 36*self.num_atoms**2

    def calculate_forces(self, E, idx):
        """
        Calulate ALL forces for first order displacement given by index
        
        Energies array of energies for all second order displacements
        """
        
        
        i1 = 6*self.num_atoms*idx
        i2 = int(6*self.num_atoms*(idx+0.5))
        i3 = 6*self.num_atoms*(idx+1)

        Em = E[i1:i2]
        Ep = E[i2:i3]
        
        return (Em-Ep)/(2*self.deltaS)
        
    def initialize_GP(self, init_samples=3, kernel=RBF()):
        """
        Initalize the Gaussian Process. 
        """
        # Data matrix and results:
        X = np.zeros((init_samples+1, self.features.shape[1]))
        y = np.zeros((init_samples+1, 1))

        # Add the minimum to the dataset:
        X[-1, :] = self.descriptor(self.atoms)
        self.atoms.set_calculator(self.calc)
        y[-1, :] = self.atoms.get_potential_energy()
        
        # Pick two and do a calculation:        
        self.feature_mask = [True for i in range(len(self.S_disp))]
        
        for i, ridx in enumerate(np.random.randint(0, len(self.S_disp), init_samples)):
            atoms = self.S_disp[ridx]; atoms.set_calculator(self.calc)
            y[i] = atoms.get_potential_energy()
            X[i, :] = self.features[ridx]
            self.feature_mask[i] = False

        # Start GP
        self.GP = Gaussian_process(X, y, kernel)
        self.GP.fit()
        
    def add_point(self):
        """
        Add a point to the GP
        """
        variance = self.GP.variance(self.features[self.feature_mask, :])
        indexs = np.argwhere(self.feature_mask)

        idx = indexs[np.argmax(variance)][0]

        atoms = self.S_disp[idx]
        atoms.set_calculator(self.calc)
        y = np.array([atoms.get_potential_energy()])
        
        self.GP.add_point(self.features[idx, :], y)
        self.GP.fit()

    def predict_energy(self):
        return self.GP.predict(self.features)
        
    def calculate_spectrum(self):
        C = np.zeros((3*self.num_atoms, 3*self.num_atoms))
        E = self.predict_energy()

        for J, i in enumerate(range(0, 6*self.num_atoms, 2)):
            fm = self.calculate_forces(E, i)
            fp = self.calculate_forces(E, i+1)
            
            C[:, J] = (fm-fp)/(2*self.deltaF)

        m = self.atoms.get_masses()**(-0.5)
        m = np.repeat(m, 3)
        w2, modes = np.linalg.eigh(m[:, None] * C * m)
        
        # Magic that turns calculated frequencies into eV (Stolen from ASE vibrations)
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        w = s * w2.astype(complex)**0.5
        return w
        
    def main(self):
        """
        Main function. Adds points to the GP until convergence of the vibrational spectrum is reached

        Note: Fake convergence criteria
        """
        t0 = dt()
        print('Starting main GP cycle')
        
        c = 3
        w = self.calculate_spectrum()
        np.save(self.folder+'w{}.npy'.format(c), w)
        while c < self.num_second:
            self.add_point()
            w = self.calculate_spectrum()
            np.save(self.folder+'w{}.npy'.format(c), w)
            c += 1
                    
        print('Main cycle finished: {} s'.format(dt()-t0))
            



    
    

        
