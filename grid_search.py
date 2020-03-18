from typing import List
import argparse, os
from joblib import Parallel, delayed

def parse_arguments(parser):
    parser.add_argument('-in','--input_dataset', type=str, default='gansan', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=True, choices=['gansan', 'disp_impact'])
    parser.add_argument('-cpu','--cpu_parallel', type=int, default=1, help='How many cpu to run in parallel to accelerate trainings', required=False)
    args = parser.parse_args()
    return args

def launch(bs: int, lr:float, ep:int):
    os.system(f"python3 reconstructor.py -ep={ep} -in={input_dataset} -lr={lr} -bs={bs} -n=\'grid-search\'")

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
# 1. Choose two HP to iterate over
# 2. Launch python3 reconstructor.py 

lrs = [1e-6,1e-7]
bses = [1024, 2048]
input_dataset = args.input_dataset


# [sqrt(i ** 2) for i in range(10)]
# Becomes
# Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

# Hence
# [[os.system(f"python3 reconstructor.py -ep=500 -in={input_dataset} -lr={lr} -bs={bs} -n=\'grid-search\'") for lr in lrs] for bs in bses]
# Becomes

Parallel(n_jobs=args.cpu_parallel)(delayed(launch)(bs, lr, 1) for lr in lrs for bs in bses)




# args = parser.parse_args()
# total_exp = len(lrs)*len(bses)
# done = 1
# for lr in lrs:
#     for bs in bses:
#         try: 
#             os.system(f"python3 reconstructor.py -ep=1 -in={input_dataset} -lr={lr} -bs={bs} -n=\'grid-search\'")
#             print(f"Experimentation with lr={lr} and batch size of {bs} is complete \n")
#             print(f"Completed {done}/{total_exp} experimentations\n")
#             print("------------------------------------------------------------------------------------------------ \n")
#         except:
#             print("Problem launching python script reconstructor.py")
#             print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n")
#         done = done+1
