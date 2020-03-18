from typing import List
import argparse, os
from joblib import Parallel, delayed

def parse_arguments(parser):
    parser.add_argument('-in','--input_dataset', type=str, default='gansan', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=True, choices=['gansan', 'disp_impact'])
    parser.add_argument('-cpu','--cpu_parallel', type=int, default=1, help='How many cpu to run in parallel to accelerate trainings', required=False)
    parser.add_argument('-ep','--epochs', type=int, help='Number of epochs to train the model.', required=True)
    args = parser.parse_args()
    return args

def launch(bs: int, lr:float, ep:int):
    os.system(f"python3 reconstructor.py -ep={ep} -in={input_dataset} -lr={lr} -bs={bs} -n=\'grid-search\'")

parser = argparse.ArgumentParser()
args = parse_arguments(parser)

lrs = [1e-6,1e-7]
bses = [1024, 2048]
input_dataset = args.input_dataset

Parallel(n_jobs=args.cpu_parallel)(delayed(launch)(bs, lr, args.epochs) for lr in lrs for bs in bses)

