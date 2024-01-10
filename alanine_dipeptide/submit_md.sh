#!/bin/bash

#SBATCH --job alanine_1ms
#SBATCH --nodes=1
#SBATCH --partition=koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -x g001,g019,g012,g013


#SBATCH --error=std.err
#SBATCH --output=std.out

#SBATCH --mail-user=ria43@pitt.edu
#SBATCH --mail-type=ALL

echo Running on `hostname`
echo workdir $PBS_O_WORKDIR

cd $SLURM_SUBMIT_DIR

#scratch drive folder to work in
SCRDIR=/scr/${SLURM_JOB_ID}

#if the scratch drive doesn't exist (it shouldn't) make it.
if [[ ! -e $SCRDIR ]]; then
        mkdir $SCRDIR
fi

chmod +rX $SCRDIR

echo scratch drive ${SCRDIR}

cp $SLURM_SUBMIT_DIR/*.pdb ${SCRDIR}
cp $SLURM_SUBMIT_DIR/*.py ${SCRDIR}

cd ${SCRDIR}

#setup to copy files back to working dir on exit
#trap "mv *.png $SLURM_SUBMIT_DIR" EXIT
trap "mv *.pdb *.dcd $SLURM_SUBMIT_DIR" EXIT
#run the MD in conda environment
#openmm is the name of my environment, yours might be struct or rdkit idk whatever
conda run -n openmm python3 md_simulation.py --pdb alanine-dipeptide.pdb --steps 1000000000000
