#!/usr/bin/env python3

#Ian objects to the following three import statements
from openmm import *
from openmm.app import *
from openmm.unit import *
from pdbfixer.pdbfixer import PDBFixer
import sys, argparse


parser = argparse.ArgumentParser(description='Simulate a PDB using OpenMM')
parser.add_argument("--pdb",required=True,help="PDB file name")
parser.add_argument("--temperature",type=float,default="300",help="Temperature for simulation in Kelvin")
parser.add_argument("--steps",type=int,default=50000000,help="Number of 2fs time steps")
parser.add_argument("--etrajectory",type=str,default="etrajectory.dcd",help="Equilibration  dcd trajectory name")
parser.add_argument("--trajectory",type=str,default="trajectory.dcd",help="Production dcd trajectory name")
parser.add_argument("--einfo",type=argparse.FileType('wt'),default=sys.stdout,help="Equilibration simulation info file")
parser.add_argument("--info",type=argparse.FileType('wt'),default=sys.stdout,help="Production simulation info file")
parser.add_argument("--system_pdb",type=str,default="system.pdb",help="PDB of system, can be used as topology")

args = parser.parse_args()

#Load PDB and add any missing residues/atoms/hydrogens (at pH 7) (like in recitation)
pdb = PDBFixer(filename=args.pdb)
#pdb.findMissingResidues()
#pdb.findMissingAtoms()
#pdb.findNonstandardResidues()
#pdb.replaceNonstandardResidues()
#pdb.addMissingAtoms()
pdb.addMissingHydrogens(7.0)

#BEFORE adding the water box, perform a minimization of the structure
#This is necessary because, as you will recall from recitation, OpenMM
#doesn't do a great job adding residues and if we wait until after adding
#water to minimize, the water will "get in the way" and prevent the minimizer
#from resolving the clashes in the modelled residues.
#Use the Amber14 forcefield with the 'implicit/gbn2.xml' water model
ff=ForceField('amber14-all.xml', 'implicit/gbn2.xml')
system = ff.createSystem(pdb.topology)
integrator = LangevinMiddleIntegrator(args.temperature*kelvin, 1/picosecond, 1*femtoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

#Using the minimized positions of the protein, add an octahedron water box
#with 1nm of padding (neutralize).
#You MUST be using the 8.0beta version of openmm so that the 
#boxShape option of addSolvent is available
#ff=ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
modeller=Modeller(pdb.topology,simulation.context.getState(getPositions=True).getPositions())
#modeller.addSolvent(ff,boxShape='octahedron',padding=1*nanometer,neutralize=True)
# Write out PDB of topology and positions of system to args.system_pdb

PDBFile.writeFile(modeller.topology, modeller.positions, open(args.system_pdb, 'w'))
# Setup the Simulation
# Note you need to add the barostat (1atm) before creating the simulation object,
# even though it should be disabled initially (interval = 0) (if you get an error with zero, you aren't using openmm8)
# PME, 1nm cutoff, HBonds constrained
# LangevinMiddleIntegrator with friction=1/ps  and 2fs timestep
pdb=PDBFile(args.system_pdb)
system = ff.createSystem(pdb.topology)
#barostat=MonteCarloBarostat(1*atmosphere, args.temperature*kelvin,0)
#system.addForce(barostat)
integrator = LangevinMiddleIntegrator(args.temperature*kelvin, 1/picosecond, 1*femtoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)


# Energy minimize
simulation.minimizeEnergy()

# Equilibrate the system in two phases for a total 70ps.  
# Ideally we would equilibrate longer (e.g. 100ps each phase), but the autograder will time out.
# We will report on the state of the system at a more fine-grained level during equilibration
# Don't change this output as it will be auto-graded

stateReporter = StateDataReporter(args.einfo, reportInterval=50,step=True,temperature=True,volume=True,potentialEnergy=True,speed=True)
dcdReporter = DCDReporter(args.etrajectory, 500)
simulation.reporters.append(stateReporter)
simulation.reporters.append(dcdReporter)

# In the first equilibration step, we gently warm up the NVT system to
# the desired temperature T. Starting at T/100, simulate 0.5ps at a time, increasing the temperature
# by T/100 every half picosecond for a total of 50ps
for i in range(100):
    integrator.setTemperature(args.temperature/100*(i+1)*kelvin)
    simulation.step(500)



# In the second equilibration step, enable the MonteCarloBarostat barostat 
#at 1atm pressure and a frequency of 25. Simulate for 20ps

#barostat.setFrequency(25)
#simulation.step(20000)



# Replace equilibration reporters with reporters that report every 10ps.
simulation.reporters = [] # append new ones
simulation.currentStep = 0
stateReporter = StateDataReporter(args.info, reportInterval=10000,step=True,temperature=True,volume=True,potentialEnergy=True,speed=True)
dcdReporter = DCDReporter(args.trajectory, 10000)
simulation.reporters.append(stateReporter)
simulation.reporters.append(dcdReporter)

# Production simulation for args.steps

simulation.step(args.steps)
