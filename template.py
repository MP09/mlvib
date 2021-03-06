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
{ir_spectrum}
$end
"""

generate_file = """RunType=GenerateQFFnMRFiles
Model=Orca
QMKeys= PBE
orcaCommand=runOrca
QFFnModes={nmodes}
dx={displacement}        
Geometry
harmonics.out
        """

compute_file = """RunType=ComputeQFFnMRFromFiles
Model=Orca
QMKeys= PBE
orcaCommand=runOrca
QFFnModes={nmodes}
dx={displacement}        
Geometry
harmonics.out
        """

vpt_file = """RunType=VPT2
Model=Orca
QMKeys= PBE
orcaCommand=runOrca
QFFnModes={nmodes}
dx={displacement}        
Geometry
harmonics.out
        """


gab="""[Gabedit Format]
[GEOCONV]
energy
{energy1}
max-force
{max_force}
rms-force
{rms_force}

[GEOMETRIES]
{geometry}
[GEOMS] 1
1 3
energy kcal/mol 1
deltaE K 1
Dipole Debye 3
{energy2}
0
-0.72790868097600 -0.96844386801360 -1.74415967988750
{geometry1}
"""

log="""===================
run {name}.ici ......
===================
dir =  /tmp/igvpt2_{name} ......
===================
Begin /tmp/igvpt2_{name}/One0
filename=/tmp/igvpt2_{name}/One0
End /tmp/igvpt2_{name}/One0
====================================================================================================================================
Only Wilke Dononelli ; wd@phys.au.dk is authorized to use this version of iGVPT2
====================================================================================================================================
seed = -1
----------------------------------------------------------
runType=ENERGY
model=ORCA
----------------------------------------------------------

Establishing connectivity : 2 connections...
Establishing connectivity : 3 connections...
Establishing connectivity : 4 connections...
Establishing connectivity : non bonded ...
#Number of free coordinates =3 
energy = -{energy_kcal}(kcal/mol) {energy_Ha}(Hartree)
Geometry saved in /tmp/igvpt2_{name}/{name}.gab file
Save molecule in /tmp/igvpt2_{name}/{name}.gab

======================"""

