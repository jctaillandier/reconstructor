from typing import List
import argparse, os

def parse_arguments(parser):
    parser.add_argument('-in','--input_dataset', type=str, default='gansan', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=True, choices=['gansan', 'disp_impact'])
    args = parser.parse_args()
    return args
    
parser = argparse.ArgumentParser()
args = parse_arguments(parser)
# 1. Choose two HP to iterate over
# 2. Launch python3 reconstructor.py 

lrs = [0.000001]
bses = [254, 512]
input_dataset = args

args = parser.parse_args()
total_exp = len(lrs)*len(bses)
done = 1
for lr in lrs:
    for bs in bses:
        try: 
            os.system(f"python3 reconstructor.py -ep=500 -in={input_dataset} -lr={lr} -bs={bs} -n=\'grid-search\'")
            print(f"Experimentation with lr={lr} and batch size of {bs} is complete \n")
            print(f"Completed {done}/{total_exp} experimentations\n")
            print("------------------------------------------------------------------------------------------------ \n")
        except:
            print("Problem launching python script reconstructor.py")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n")
        done = done+1
