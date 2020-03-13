from typing import List
import argparse, os

parser = argparse.ArgumentParser()
# 1. Choose two HP to iterate over
# 2. Launch python3 reconstructor.py(Hp1, Hp2)

lrs = [0.000025, 0.00001, 0.0000075, 0.000005, 0.000001]
bses = [64, 128, 254, 512]
input_dataset = 'disp_impact'

args = parser.parse_args()
total_exp = len(lrs)*len(bses)
done = 1
for lr in lrs:
    for bs in bses:
        try: 
            os.system(f"python3 reconstructor.py -ep=100 -in={input_dataset} -lr={lr} -bs={bs} -n=\'grid-search\'")
            print(f"Experimentation with lr={lr} and batch size of {bs} is complete \n")
            print(f"Completed {done}/{total_exp} experimentations\n")
            print("------------------------------------------------------------------------------------------------ \n")
        except:
            print("Problem launching python script reconstructor.py")
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n")
        done = done+1
