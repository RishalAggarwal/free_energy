#!/bin/bash    
#SBATCH --job-name=vscode
#SBATCH -p koes_gpu    
#SBATCH -x g017
#SBATCH --constraint L40
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1    
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=1    
#SBATCH --output vscode.stdout    

#kill vs code server when this job finishes
trap "ps uxa | grep .vscode-server | awk '{print \$2}' | xargs kill" EXIT

#if we are going to use a GPU, we need to know which one was assigned
nvidia-smi
#do nothing - our sole purpose is to enable ssh to node
while true
do
    sleep 10
done
