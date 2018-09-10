from ase.build import molecule
from ase.collections import g2
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import Trajectory, read, write

def make_tests():
    """
    Takes molecules fron the G2-database and relaxes with the built in ASE EMT calculator.
    Molecules containing elements not described by the EMT calculator are discarded.

    Generated structures are only ment for testing numerics.
    """

    for name in g2.names:
        atoms = molecule(name)
        try:
            atoms.set_calculator(EMT())
            BFGS(atoms, logfile=None).run(fmax=0.01)
            traj = Trajectory(name+'.traj', mode='w')
            traj.write(atoms)
        except NotImplementedError:
            pass

