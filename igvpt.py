import numpy as np
import ase.units as units
#from vibration_analysis import Vibration_analysis
from mlvib.template import template, generate_file, gab, compute_file, log, vpt_file
from mlvib.calculate_trajectory import calculate_trajectory
from mlvib.calculate_specifics import calculate_specifics
from mlvib.gaussian_process import Gaussian_process
from ase.io import Trajectory, read, write
from ase.db import connect

from time import sleep
import pickle
import subprocess
import glob
import os
import re

eV_to_kcal = 23.06055 
eV_to_Ha = 0.0367493
A_to_a0 = 1.88972613
force_con = eV_to_Ha/A_to_a0
hess_con = eV_to_Ha/A_to_a0**2
dipole_con = 1/(1.602176*10**(-18))
eV_to_cm = 8065.544

class IGVPT:

    def __init__(self, atoms, calc, QFF_calc=None, name='test', delta=0.01, nmodes=2):
        """
        ASE interface to iGVPT2 program. 

        iGVPT2 requires several steps:

        1. Calculation of normal modes, harmonic frequencies and dipole derivatives. 
        2. Generation/calculation of displaced structures to obtain cubic and quartic force constants
        3. Evaluation of anharmonic frequencies.

        Inputs:
        - atoms: ASE atoms object (relaxed).
        """

        # Atoms object:
        self.atoms = atoms
        self.num_atoms = len(atoms)
        self.symbols = self.atoms.get_chemical_symbols()
        self.num_normals = 3*self.num_atoms-6

        # Settings:
        self.delta = 0.01
        self.displacement = 0.01
        self.nmodes = nmodes

        if self.nmodes == 1:
            self.num_extra = 1+2*self.num_normals
        elif self.nmodes == 2:
            self.num_extra = 1+6*self.num_normals**2
        elif self.nmodes == 3:
            self.num_extra = (1+6*self.num_normals**2 + 8*self.num_normals*(self.num_normals-1)
                              *(self.num_normals-2)//6)
        elif self.nmodes == 4:
            self.num_extra = (1 + 6*self.num_normals**2 + 8*self.num_normals*(self.num_normals-1)
                              *(self.num_normals-2)//6)
        
        # Naming scheme:
        self.name = name
        self.harm_folder = 'harmonics/'
        self.ici_folder = 'ici/'
        self.gab_folder = 'gab/'
        self.QFF_folder = 'QFF/'
        for folder in [self.harm_folder, self.ici_folder, self.gab_folder, self.QFF_folder]:
            if not os.path.exists(folder):
                os.mkdir(folder)
        self.top_folder = os.getcwd()

        # Harmonic information
        self.gs_energy = None
        self.gs_gradient = None
        self.hessian = None
        self.normal_modes = None
        self.dipole_derivative = None
        self.harm_freq = None

        # Anharmonic information:
        self.QFF_atoms = []
        self.history = []

        # Calculator:
        self.calc = calc

        if QFF_calc is None:
            self.QFF_calc = self.calc
        else:
            self.QFF_calc = QFF_calc

    def calculate_displacements(self):
        """
        Create trajectory object with displaced atoms for normal mode calculation.

        Order will be:
        1. 0, x, +
        2. 0, x, -
        3. 0, y, +
        etc..
        
        """
        trajectory = Trajectory(self.name+'_disp.traj', mode='w')
        trajectory.write(self.atoms)
        atoms = self.atoms.copy()
        for i in range(self.num_atoms):
            for d in range(3):
                atoms.positions[i, d] += self.delta
                trajectory.write(atoms)
                atoms.positions[i, d] -= 2*self.delta
                trajectory.write(atoms)
                atoms.positions[i, d] += self.delta

    def calculate_harmonics(self):
        """
        Start calculations required to get the harmonics/normal modes.
        """
        os.chdir(self.harm_folder)
        self.calculate_displacements()

        with open('calc.pckl', 'wb') as pickle_file:
            pickle.dump(self.calc, pickle_file)
        
        num_calcs = 6*self.num_atoms+1
        if not os.path.exists('out/'):
            os.mkdir('out/')
        calculate_trajectory(self.name+'_disp.traj', num_calcs, self.name)
        subprocess.run('sbatch job_file.sh', shell=True)
        subprocess.run('echo "running" > .checkpoint', shell=True)
        os.chdir(self.top_folder)

    def gather_files(self):
        """
        Gather atom objects from 
        """
        os.chdir(self.harm_folder)
        traj = Trajectory(self.name+'_harm.traj', mode='w')
        for j in range(6*self.num_atoms+1):
            traj.write(read('job_{}/atoms.traj'.format(j)))
        os.chdir(self.top_folder)


    def calculate_hessian(self):
        """
        Calculate the ground-state Hessian. 
        """
        atoms = read(self.harm_folder+self.name+'_harm.traj', index=':')
        self.hessian = np.zeros((3*self.num_atoms, 3*self.num_atoms))
        self.gs_energy = atoms[0].get_potential_energy()
        self.gs_gradient = atoms[0].get_forces()

        for J, j in enumerate(range(1, len(atoms), 2)):
            fp = atoms[j].get_forces()
            fm = atoms[j+1].get_forces()
            self.hessian[:, J] = (fm-fp).ravel()/(2*self.delta)
            
        m = self.atoms.get_masses()**(-0.5)
        m = np.repeat(m, 3)                                                                          
        w2, modes = np.linalg.eigh(m[:, None] * self.hessian * m)
        modes = modes.T
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        self.normal_modes = -modes
        self.harm_freq = s*w2.astype(complex)**0.5

        self.normal_modes[0:6, :] = 0.0

    def calculate_dipole_derivative(self):
        """
        Calculate the dipole derivatives.
        """
        atoms = read(self.harm_folder+self.name+'_harm.traj', index=':')
        
        self.dipole_derivative = np.zeros((3*self.num_atoms, 3))
        
        for J, j in enumerate(range(1, len(atoms), 2)):
            dp = atoms[j].get_dipole_moment()
            dm = atoms[j+1].get_dipole_moment()
            self.dipole_derivative[J, :] = (dp-dm)/(2*self.delta)
        
        
    def generate_QFF_files(self):
        """
        Write 'generate' ici file.
        """
        settings = {'displacement':self.displacement, 'nmodes':self.nmodes}
        os.chdir(self.ici_folder)
        with open(self.name+'.ici', 'w') as f:
            print(generate_file.format(**settings), file=f)
        subprocess.run('/home/machri/iGVPT2/bin/igvpt2 ' + self.name+'.ici > out.out', shell=True)
        os.chdir(self.top_folder)


    def generate_QFF(self):
        """
        Write 'compute' ici file and run igvpt2
        """
        settings = {'displacement':self.displacement, 'nmodes':self.nmodes}
        os.chdir(self.gab_folder)
        with open(self.name+'.ici', 'w') as f:
            print(compute_file.format(**settings), file=f)

        # Generate txt file:
        subprocess.run('/home/machri/iGVPT2/bin/igvpt2 ' + self.name+'.ici > out.out', shell=True)
        subprocess.run("sed -i '/RunType/d' ./{}QFF.txt".format(self.name), shell=True)
        subprocess.run("echo RunType=VPT2 > {}QFF.ici".format(self.name), shell=True)
        subprocess.run("cat {}QFF.txt >> {}QFF.ici".format(self.name, self.name), shell=True)
        subprocess.run('/home/machri/iGVPT2/bin/igvpt2 ' + self.name+'QFF.ici > ../results.out',
                       shell=True)
        
        os.chdir(self.top_folder)
        
    def read_ici(self):        
        self.num_extra = len(glob.glob(self.ici_folder + self.name +'QFF_*'))
        
        traj = Trajectory(self.QFF_folder + self.name + 'QFF.traj', mode='w')
        for num in range(0, self.num_extra):
            f = self.ici_folder + self.name + 'QFF_{}.ici'.format(num)
            with open(f, 'r') as open_file:
                prev = 'kage'
                pos = np.zeros((self.num_atoms, 3))
                i = 0
                for sz, line in enumerate(open_file.readlines()):
                    if line == 'Geometry\n':
                        prev = 'geo'
                        continue
                    if prev == 'geo':
                        prev = 'numbers'
                        continue
                    if prev == 'numbers':
                        pos[i] = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)[4:7]]
                        i += 1
            atoms = self.atoms.copy()
            atoms.set_positions(pos)
            self.QFF_atoms.append(atoms)
            traj.write(atoms)
        
    def calculate_QFF(self):
        """
        Queue all calculations required for QFF evaluation.
        Used in conjuction with gather_QFF if a 'real' calculator is used for self.QFF_calc
        """
        os.chdir(self.QFF_folder)
        with open('calc.pckl', 'wb') as pickle_file:
            pickle.dump(self.QFF_calc, pickle_file)
        
        num_calcs = len(self.QFF_atoms)
        calculate_trajectory(self.name+'QFF.traj', num_calcs, self.name)
        subprocess.run('sbatch job_file.sh', shell=True)
        subprocess.run('echo "running" > .checkpoint', shell=True)
        os.chdir(self.top_folder)

    def gather_QFF(self):
        """
        Gather calculations files for QFF evaluation. 
        """
        self.num_extra = len(glob.glob(self.ici_folder + self.name +'QFF_*'))
        traj = Trajectory(self.QFF_folder+self.name+'QFF_calced.traj', mode='w')
        for j in range(self.num_extra):
            traj.write(read(self.QFF_folder+'job_{}/atoms.traj'.format(j)))
        self.QFF_atoms = read(self.QFF_folder+self.name+'QFF_calced.traj', index=':')
        

    def write_gabs(self, use_atoms=True):
        """
        Write gab file for displaced atoms. 
        """

        if use_atoms:
            self.E_QFF = np.array([atoms.get_potential_energy() for atoms in self.QFF_atoms])
            self.F_QFF = np.array([atoms.get_forces() for atoms in self.QFF_atoms])
        
        subprocess.run('cp {}/*QFF_*.ici {}/.'.format(self.ici_folder, self.gab_folder), shell=True)
        for idx, atoms in enumerate(self.QFF_atoms):
            settings = {}
            settings['energy1'] = '{:.6f}'.format(self.E_QFF[idx]*eV_to_kcal)
            settings['energy2'] = '{:.14f}'.format(self.E_QFF[idx]*eV_to_kcal)
            forces = self.F_QFF[idx]
            settings['max_force'] = np.max(np.abs(forces.flatten()))*eV_to_Ha/A_to_a0
            settings['rms_force'] = np.sqrt(np.mean(forces.flatten()**2))*eV_to_Ha/A_to_a0
            
            geometry = '{}\n\n'.format(self.num_atoms)
            geometry1 = '{} 0 1\n'.format(self.num_atoms)
            for i, atom in enumerate(atoms):
                geometry += ' {}'.format(atom.symbol)
                geometry1 += ' {}'.format(atom.symbol)*4 + ' 0 34 2 1'
                for coordinate in atom.position:
                    geometry += ' {:.8f}'.format(coordinate*A_to_a0)
                    geometry1 += ' {:.12f}'.format(coordinate*A_to_a0)
                geometry1 += ' 0'
                #geometry1 += ' 0 0 0 GRADIENT'
                #for force in forces[i, :]:
                #    geometry1 += ' {:.12f}'.format(force*eV_to_Ha/A_to_a0)
                geometry += '\n'
                geometry1 += '\n'
                settings['geometry'] = geometry
                settings['geometry1'] = geometry1
        
            with open(self.gab_folder+self.name+'QFF_{}.gab'.format(idx), 'w') as f:
                print(gab.format(**settings), file=f)

            settings = {}
            settings['energy_kcal'] = self.E_QFF[idx]*eV_to_kcal
            settings['energy_Ha'] = self.E_QFF[idx]*eV_to_Ha
            settings['name'] = self.name + 'QFF_{}.gab'.format(idx)
            with open(self.gab_folder + self.name+'QFF_{}.log'.format(idx), 'w') as f:
                print(log.format(**settings), file=f)

    def write_harmonics(self):
        """
        Write iGVPT2 input file in the style of Orca Hessian file (template.txt). 
                
        Note: Double check units
        """
                
        # Load template:
        infile = template

        # Stuff to fill in:
        keys = ['num_atoms', 'energy', 'gradient', 'pos', 'hessian', 'vib',
                'normal_modes', 'pos_mass', 'dipole', 'ir_spectrum']
        fill_dict = {key:'' for key in keys}

        # Easy ones:
        fill_dict['num_atoms'] = ' {}'.format(self.num_atoms)
        fill_dict['energy'] = '    {}'.format(self.gs_energy*eV_to_Ha)

        # Gradient:
        grad = ''
        for row in self.gs_gradient:
            for num in row:
                grad += 7*' '+'{:1.12f}\n'.format(num*force_con)
        fill_dict['gradient'] = grad[0:-2]

        # Hessian:
        num_sections = 3*self.num_atoms//5+1
        hes = '{}\n'.format(3*self.num_atoms)
        for sect in range(num_sections):
            if sect == 0:
                add = 1
                hes += 20*' ' + '{}'.format(sect)
            else:
                add = 0
            for col in range(sect*5+add, min((sect+1)*5, self.num_atoms**2)):
                hes += 18*' ' + '{}'.format(col) 
            hes += '\n'
            for i in range(3*self.num_atoms):
                if sect > 0:
                    hes += ' '
                hes += '    {}   '.format(i)
                for j in range(sect*5, min((sect+1)*5, 3*self.num_atoms)):
                    if self.hessian[i, j] > 0:
                        hes += '   {:1.10E}'.format(self.hessian[i, j]*hess_con)
                    else:
                        hes += '  {:1.10E}'.format(self.hessian[i, j]*hess_con)
                hes += '\n'
        fill_dict['hessian'] = hes

        # Normal modes:
        num_sections = self.num_atoms**2//5+1
        norm = '{} {}\n'.format(3*self.num_atoms, 3*self.num_atoms)
        for sect in range(num_sections):
            if sect == 0:
                norm += 20*' ' + '{}'.format(sect)
                add = 1
            else:
                add = 0
            for col in range(sect*5+add, min((sect+1)*5, 3**self.num_atoms)):
                norm += 18*' ' + '{}'.format(col) 
            norm += '\n'
            for i in range(3*self.num_atoms):
                if sect > 0:
                    norm += ' '
                norm += '    {}   '.format(i)
                for j in range(sect*5, min((sect+1)*5, 3*self.num_atoms)):
                    if self.normal_modes[j, i] > 0:
                        norm += '   {:1.10E}'.format(self.normal_modes[j, i])
                    else:
                        norm += '  {:1.10E}'.format(self.normal_modes[j, i])
                norm += '\n'
        fill_dict['normal_modes'] = norm

        # Position 1:
        pos = ''
        pos1 = '{}\n'.format(self.num_atoms)
        for atom in self.atoms:
            pos += '   {}'.format(atom.number)
            pos1 += ' {}     {}'.format(atom.symbol, atom.mass)
            for i in range(3):
                if atom.position[i] > 0:
                    pos += '     {:.7f}'.format(atom.position[i]*A_to_a0)
                    pos1 += '      {:.7f}'.format(atom.position[i]*A_to_a0)
                else:
                    pos += '    {:.7f}'.format(atom.position[i]*A_to_a0)
                    pos1 += '     {:.7f}'.format(atom.position[i]*A_to_a0)
            pos += '\n'
            pos1 += '\n'
        fill_dict['pos'] = pos
        fill_dict['pos_mass'] = pos1

        # Dipole derivative:
        dpd = '{}\n'.format(3*self.num_atoms)
        for i in range(3*self.num_atoms):
            for j in range(3):
                if self.dipole_derivative[i, j] > 0:
                    dpd += '    {:.10E}'.format(self.dipole_derivative[i, j]*dipole_con)
                else:
                    dpd += '   {:.10E}'.format(self.dipole_derivative[i, j]*dipole_con)
            dpd += '\n'
        fill_dict['dipole'] = dpd

        # Harmonic frequencies:
        harm = '{}\n'.format(self.num_atoms)
        for jj, freq in enumerate(self.harm_freq):
            if freq*eV_to_cm < 200:
                freq = 0.0
            harm += '    {}     {:.6f}\n'.format(jj, freq.real*eV_to_cm)
        fill_dict['vib'] = harm

        # IR Spectrum:
        ir = '{}\n'.format(3*self.num_atoms)
        for i in range(3*self.num_atoms):
            if self.harm_freq[i]*eV_to_cm < 200:
                self.harm_freq[i] = 0.0
        
            for j in range(5):
                if j == 0:
                    s = len('{:.2f}'.format(self.harm_freq[i].real*eV_to_cm))-3
                    ir += (6-s)*' ' +'{:.2f}'.format(self.harm_freq[i].real*eV_to_cm)
                else:
                    ir += '       0.000'
            ir += '\n'
        fill_dict['ir_spectrum'] = ir

        for folder in [self.ici_folder, self.gab_folder]:
            with open(folder+'harmonics.out', 'w') as f:
                print(infile.format(**fill_dict), file=f)

    def summarize_results(self, print_summary=True):
        """
        Read output file from igvpt2
        """
        
        with open('results.out') as f:
            lines = f.readlines()
        start = False
        startsoon = False
        mode_num = 0

        fundamentals = []
        for line in lines:
            if 'Fundamental Bands' in line:
                startsoon = True
                start_in = -4
            if startsoon == True:
                start_in += 1
                if start_in >= 0:
                    start = True
                    startsoon = False
                    if print_summary:
                        print('| # |  Harmonic  |  Fundamental  |   Delta   |')
                    
            elif start:

                if mode_num < (3*self.num_atoms-6):
                
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                    harm = float(nums[0])
                    anharm = float(nums[1])
                    diff = float(anharm-harm)
                    fundamentals.append(anharm)
                    if print_summary:
                        print('| {} |   {:.2f}  |    {:.2f}    |  {:.2f}  |'.
                              format(mode_num, harm, anharm, diff))
                    mode_num += 1
                    
                    if 'Overtones' in line:
                        break
        return fundamentals
                
            

    def calculate_anharmonics(self):
        """
        Function for performing a full anharmonic calculation with specified calculator.

        Does the following:
  
        1. Calculates harmonics.
        2. Calculates QFF
        3. Evaluates anharmomics
        """
        if not os.path.exists(self.harm_folder + '.checkpoint'):
            print('Starting calculations for harmonics..')
            self.calculate_harmonics()
            
        elif not os.path.exists(self.QFF_folder + '.checkpoint'):
            print('Starting calculations for QFF')
            self.gather_files()
            self.calculate_hessian()
            self.calculate_dipole_derivative()
            self.write_harmonics()
            self.generate_QFF_files()
            self.read_ici()
            self.calculate_QFF()
        else:
            print('Finalizing..')
            self.gather_QFF()
            self.write_gabs()
            self.generate_QFF()
            self.summarize_results()

    def approximate_anharmonics(self, descriptor, kernel):
        """
        Main function for calculating anharmonics by approxixmating the PES with a gaussian process. 

        Steps:
        1. Calculates the harmonics with specified calculator (self.calc)
        2. Generates geometries for anharmonics, calculate feature vectors for all geometries
        3. ??

        Parameters:
        -- descriptor: global descriptor object 
        """

        if not os.path.exists(self.harm_folder + '.checkpoint'):
            print('Starting calculations for harmonics..')
            self.calculate_harmonics()
            
        elif not os.path.exists(self.QFF_folder + '.checkpoint'):
            print('Generating geometries for QFF')
            self.gather_files()                 # Gather files for Hessian
            self.calculate_hessian()            # Calculate Hessian
            self.calculate_dipole_derivative()  # Calculate dipole derivative
            self.write_harmonics()              # Write harmonics files
            self.generate_QFF_files()           # Generate QFF ici files
            self.read_ici()                     # Read geometries for ici files

            # Calculate the feature of all harmonic displacements:
            harm_atoms = read(self.harm_folder + self.name+'_harm.traj', index=':')
            feature = descriptor(harm_atoms[0])
            harm_features = np.zeros((len(harm_atoms), feature.shape[0]))
            harm_features[0] = feature
            for i, atoms in enumerate(harm_atoms[1::]):
                harm_features[i+1, :] = descriptor(atoms)
            E_harm = np.array([atoms.get_potential_energy() for atoms in harm_atoms]).reshape(-1, 1)

            # Intialize Gaussian Process with harmonic calculations:
            self.GP = Gaussian_process(harm_features, E_harm, kernel)
            self.GP.fit()
            
            # Calculate features of all QFF displacements:
            self.features = np.zeros((len(self.QFF_atoms), feature.shape[0]))
            for i, atoms in enumerate(self.QFF_atoms):
                self.features[i, :] = descriptor(atoms)

            self.E_QFF = self.GP.predict(self.features)
            self.F_QFF = np.zeros((self.num_extra, len(atoms), 3))

            self.write_gabs(use_atoms=False)
            self.generate_QFF()
            self.history.append(self.summarize_results())

            # Start adding calculations:
            self.concurrent_calculations = 2
            self.time_between_checks = 5

            remainder = self.num_extra
            index_mask = np.array([False for i in range(self.num_extra)])
            while remainder > 0:

                # Pick the next configuration(s) to calculate:

                # Variance based:
                #variance = self.GP.variance(self.features)
                #temp_idxs = np.argsort(-variance)[0:self.concurrent_calculations]
                
                # Random:
                temp_idxs = np.random.choice(np.argwhere(index_mask==False).reshape(-1),
                            self.concurrent_calculations)

                # Check if the calculations have be done previously:
                idxs = []
                for i, idx in enumerate(temp_idxs):
                    if not os.path.exists(self.QFF_folder + 'job_{}/.finished'.format(idx)):
                        idxs.append(idx)
                idxs = np.array(idxs)
                print(idxs)
                
                if idxs.size:
                    os.chdir(self.QFF_folder)
                    with open('calc.pckl', 'wb') as pickle_file:
                        pickle.dump(self.QFF_calc, pickle_file)
                        calculate_specifics(idxs, self.name+'QFF.traj', self.concurrent_calculations,
                                            self.name)
                        subprocess.run('sbatch job_file.sh', shell=True)
                    os.chdir(self.top_folder)

                idxs = temp_idxs.copy()
                # Wait for them ...
                calculation_running = True
                while calculation_running:
                    calcs_running = 0
                    for idx in idxs:
                        if not os.path.exists(self.QFF_folder + 'job_{}/.finished'.format(idx)):
                            calcs_running += 1
                    if calcs_running == 0:
                        calculation_running = False
                    else:
                        sleep(self.time_between_checks)

                # When finished add them to GP
                E = np.zeros(self.concurrent_calculations)
                for i, idx in enumerate(idxs):
                    atoms = read(self.QFF_folder + 'job_{}/atoms.traj'.format(idx))
                    E[i] = atoms.get_potential_energy()
                    index_mask[idx] = True
                    
                self.GP.add_point(self.features[idxs, :], E)
                self.GP.fit()

                # Write new gab files:
                self.E_QFF = self.GP.predict(self.features)
                self.write_gabs(use_atoms=False)
                self.generate_QFF()
                self.history.append(self.summarize_results(print_summary=True))

                remainder -= self.concurrent_calculations

                remaining = np.argwhere(index_mask == False).reshape(-1)
                
                

                
                
                
        
