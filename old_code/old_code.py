    def approximate_anharmonics(self, descriptor, kernel, print_summary=False, use_gs_prior=True,
                                normalize=True):
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
            #feature = descriptor(harm_atoms[0])
            feature = descriptor.get_features(harm_atoms[0])
            harm_features = np.zeros((len(harm_atoms), feature.shape[1]))
            harm_features[0] = feature
            for i, atoms in enumerate(harm_atoms[1::]):
                #harm_features[i+1, :] = descriptor(atoms)
                harm_features[i+1, :] = descriptor.get_features(atoms)
            E_harm = np.array([atoms.get_potential_energy() for atoms in harm_atoms]).reshape(-1, 1)

            if normalize:
                harm_features = self.normalize_features(harm_features)

            # Dump the calculator:
            with open(self.QFF_folder+'calc.pckl', 'wb') as pickle_file:
                pickle.dump(self.QFF_calc, pickle_file)
            
            # Add the groundstate as the prior:
            if use_gs_prior:
                # Calculate if required:
                self.calculate_and_wait([0])

                # Find calculation:
                atoms = read(self.QFF_folder + 'job_0/atoms.traj')
                prior = atoms.get_potential_energy()
            else:
                prior = 0
                
            # Intialize Gaussian Process with harmonic calculations:
            self.GP = Gaussian_process(harm_features, E_harm, kernel, prior=prior)
            self.GP.fit()
            
            # Calculate features of all QFF displacements:
            self.features = np.zeros((len(self.QFF_atoms), feature.shape[1]))
            for i, atoms in enumerate(self.QFF_atoms):
                #self.features[i, :] = descriptor(atoms)
                self.features[i, :] = descriptor.get_features(atoms)

            if normalize:
                self.features = self.normalize_features(self.features)

            self.E_QFF = self.GP.predict(self.features)
            self.F_QFF = np.zeros((self.num_extra, len(atoms), 3))

            self.write_gabs(use_atoms=False)
            self.generate_QFF()
            self.history.append(self.summarize_results(print_summary=True))

            # Start adding calculations:
            self.concurrent_calculations = 1
            self.time_between_checks = 5

            remainder = self.num_extra
            index_mask = np.array([False for i in range(self.num_extra)])
            count = 0

            print('Got this far')
            
            print(remainder)
            while remainder > 0:
                # Pick the next configuration(s) to calculate:
                print('#'*40)
                print('Remainder: {}'.format(remainder))
                
                # Variance based:
                variance = self.GP.variance(self.features)
                idxs = np.argsort(-variance)[0:self.concurrent_calculations]
                
                # Random:
                #idxs = np.random.choice(np.argwhere(index_mask==False).reshape(-1),
                #            self.concurrent_calculations)
                
                self.calculate_and_wait(idxs, self.time_between_checks)
                
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
                self.history.append(self.summarize_results(print_summary=print_summary))

                remainder -= self.concurrent_calculations

                remaining = np.argwhere(index_mask == False).reshape(-1)
                count += 1

            self.history = np.array(self.history)

            # Save some important stuff:
            np.save(self.ML_folder + 'history.npy', self.history)

            with open(self.ML_folder + 'GP.pckl', 'wb') as pickle_file:
                pickle.dump(self.GP, pickle_file)

            with open(self.ML_folder + 'descriptor.pckl', 'wb') as pickle_file:
                pickle.dump(descriptor, pickle_file)
