template = """
#
# Number of atoms
#
{num_atoms}
#
# The current total energy in Eh
#
{energy}
#
# The current gradient in Eh/bohr
#
{gradient}     
#
# The atomic numbers and current coordinates in Bohr
#
{pos}

$orca_hessian_file

$act_atom
  0

$act_coord
  0

$act_energy
        0.000000

$hessian
{hessian}

$vibrational_frequencies
{vib}
$normal_modes
{normal_modes}
#
# The atoms: label  mass x y z (in bohrs)
#
$atoms
{pos_mass}
$actual_temperature
  0.000000

$dipole_derivatives
{dipole}

#
# The IR spectrum
#  wavenumber T**2 TX TY  TY
#
$ir_spectrum
9
      0.00       0.0000       0.0000       0.0000       0.0000
      0.00       0.0000       0.0000       0.0000       0.0000
      0.00       0.0000       0.0000       0.0000       0.0000
      0.00       0.0000       0.0000       0.0000       0.0000
      0.00       0.0000       0.0000       0.0000       0.0000
      0.00       0.0000       0.0000       0.0000       0.0000
   1608.78      47.4887       4.8722       4.8722      -0.1123
   3688.90       2.2236       1.0543       1.0543      -0.0243
   3787.88      19.7475      -3.1423       3.1423      -0.0000

$end
"""

