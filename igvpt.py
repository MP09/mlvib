import numpy as np
from vibration_analysis import Vibration_analysis
from template import template

eV_to_Ha = 0.0367493

class IGVPT:

    def __init__(self, atoms):
        """
        ASE interface to iGVPT2 program. 

        iGVPT2 requires several steps:

        1. Calculation of normal modes, harmonic frequencies and dipole derivatives. 
        2. Generation/calculation of displaced structures to obtain cubic and quartic force constants.
        3. Evaluation of anharmonic frequencies.

        Inputs:
        - atoms: ASE atoms object (relaxed).
        """

        self.atoms = atoms
        self.num_atoms = len(atoms)

    def get_hessian(self):
        """
        Get the Hessian from a 'Vibrational_analysis' object'. 
        """
        self.hessian = self.vib.get_hessian()

    def get_dipole_derivatives(self):
        """
        Calculate the deriative of the dipole moment according to atomic displacements.
        """

    def write_input(self):
        """
        Write iGVPT2 input file in the style of Orca Hessian file (template.txt). 

        Note: currently only writes stuff in whatever unit it is in, need to convert some stuff.
        """
        
        # Load template:
        infile = template

        # Stuff to fill in:
        keys = ['num_atoms', 'energy', 'gradient', 'pos', 'hessian', 'vib',
                'normal_modes', 'pos_mass', 'dipole']
        fill_dict = {key:'' for key in keys}

        # Easy ones:
        fill_dict['num_atoms'] = ' {}'.format(self.num_atoms)
        fill_dict['energy'] = '    {}'.format(self.gs_energy)

        # Gradient:
        grad = ''
        for row in self.gs_gradient:
            for num in row:
                grad += 7*' '+'{:1.12f}\n'.format(num)
        fill_dict['gradient'] = grad[0:-2]

        # Hessian:
        num_sections = self.num_atoms**2//5+1
        hes = '{}\n'.format(self.num_atoms**2)
        for sect in range(num_sections):
            hes += 20*' ' + '{}'.format(sect)
            for col in range(sect*5+1, min((sect+1)*5, self.num_atoms**2)):
                hes += 18*' ' + '{}'.format(col) 
            hes += '\n'
            for i in range(self.num_atoms**2):
                if sect > 0:
                    hes += ' '
                hes += '    {}   '.format(i)
                for j in range(sect*5, min((sect+1)*5, self.num_atoms**2)):
                    if self.hessian[i, j] > 0:
                        hes += '   {:1.10E}'.format(self.hessian[i, j])
                    else:
                        hes += '  {:1.10E}'.format(self.hessian[i, j])
                hes += '\n'
        fill_dict['hessian'] = hes

        # Normal modes:
        num_sections = self.num_atoms**2//5+1
        norm = '{}\n'.format(self.num_atoms**2)
        for sect in range(num_sections):
            norm += 20*' ' + '{}'.format(sect)
            for col in range(sect*5+1, min((sect+1)*5, self.num_atoms**2)):
                norm += 18*' ' + '{}'.format(col) 
            norm += '\n'
            for i in range(self.num_atoms**2):
                if sect > 0:
                    norm += ' '
                norm += '    {}   '.format(i)
                for j in range(sect*5, min((sect+1)*5, self.num_atoms**2)):
                    if self.normal_modes[i, j] > 0:
                        norm += '   {:1.10E}'.format(self.normal_modes[i, j])
                    else:
                        norm += '  {:1.10E}'.format(self.normal_modes[i, j])
                norm += '\n'
        fill_dict['normal_modes'] = norm

        # Position 1:
        pos = ''
        pos1 = ''
        for atom in self.atoms:
            pos += '   {}'.format(atom.number)
            pos1 += ' {}     {}'.format(atom.symbol, atom.mass)
            for i in range(3):
                if atom.position[i] > 0:
                    pos += '     {:.7f}'.format(atom.position[i])
                    pos1 += '      {:.7f}'.format(atom.position[i])
                else:
                    pos += '    {:.7f}'.format(atom.position[i])
                    pos1 += '     {:.7f}'.format(atom.position[i])
            pos += '\n'
            pos1 += '\n'
        fill_dict['pos'] = pos
        fill_dict['pos_mass'] = pos1

        # Dipole derivative:
        dpd = '{}\n'.format(3*self.num_atoms)
        for i in range(3*self.num_atoms):
            for j in range(3):
                if self.dipole_derivative[i, j] > 0:
                    dpd += '    {:.10E}'.format(self.dipole_derivative[i, j])
                else:
                    dpd += '   {:.10E}'.format(self.dipole_derivative[i, j])
            dpd += '\n'
        fill_dict['dipole'] = dpd

        # Harmonic frequencies:
        harm = '{}\n'.format(self.num_atoms)
        for jj, freq in enumerate(self.harm_freq):
            harm += '    {}     {:.6f}\n'.format(jj, freq)
        fill_dict['vib'] = harm
        



        print(infile.format(**fill_dict))
        
        

        
        


if __name__ == '__main__':
    from test_systems.LJ38.factory import factory

    n = 3
    atoms = factory(n=n)
    atoms.positions[0, 0] = -1
    
    A = IGVPT(atoms)
    A.gs_energy = -10
    A.gs_gradient = np.random.rand(9).reshape(3, 3)
    A.hessian = np.random.rand(9*9).reshape(9, 9)
    #A.hessian[0, 0] = -A.hessian[0, 0]
    mask = np.random.rand(9*9).reshape(9, 9) > 0.5
    A.hessian[mask] = -A.hessian[mask]
    A.normal_modes = np.random.rand(9*9).reshape(9, 9)
    A.dipole_derivative = np.random.rand(3*n*3).reshape(3*n, 3)
    mask = np.random.rand(3*n*3).reshape(3*n, 3) > 0.5
    A.dipole_derivative[mask] = -A.dipole_derivative[mask]

    A.harm_freq = np.random.rand(n**2)
    
    

    A.write_input()
    



    
        

        
