#!/bin/bash
#SBATCH --account=def-gambsseb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120000M
#SBATCH --time=24:00:00
#SBATCH --job-name=100_epochs_a=0
#SBATCH --mail-user=jean-christophe.taillandier@umontreal.ca
#SBATCH --output=./%x-%j.out
#SBATCH --mail-type=ALL
source ~/gpu_ready_pyenv/bin/activate
python3 ./reconstructor/reconstructor.py params.yaml '100ep_a=0'
