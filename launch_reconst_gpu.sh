#!/bin/bash
#SBATCH --account=def-gambsseb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50000M
#SBATCH --time=06:00:00
#SBATCH --job-name=adult
#SBATCH --mail-user=jean-christophe.taillandier@umontreal.ca
#SBATCH --output=./%x-%j.out
#SBATCH --mail-type=ALL
source ~/gpu_ready_pyenv/bin/activate
python3 ./reconstructor.py -bs 4096 -ep 5000 -in=gansan -a=0.9875
