#!/bin/bash
#SBATCH --account=def-gambsseb
#SBATCH --cpus-per-task=1
#SBATCH --mem=50000M
#SBATCH --time=06:00:00
#SBATCH --job-name=adult_2048
#SBATCH --mail-user=jean-christophe.taillandier@umontreal.ca
#SBATCH --output=./%x-%j.out
#SBATCH --mail-type=ALL
source ~/gpu_ready_pyenv/bin/activate
python3 ./reconstructor.py -n adult -bs 2048 -ep 5000 -dr='tsne'
