from typing import List
import argparse, os
from joblib import Parallel, delayed
import smtplib

def parse_arguments(parser):
    parser.add_argument('-in','--input_dataset', type=str, default='gansan', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=False, choices=['gansan', 'disp_impact'])
    parser.add_argument('-cpu','--cpu_parallel', type=int, default=1, help='How many cpu to run in parallel to accelerate trainings', required=False)
    parser.add_argument('-a','--alpha', type=float, default=0.9875, help='Value of alpha used when  sanitizing the dataset we use as input.', required=False, choices=[0,0.25,0.8,0.9875])
    parser.add_argument('-ep','--epochs', type=int, help='Number of epochs to train the model.', required=True)
    args = parser.parse_args()
    return args

def launch(bs: int, lr:float, ep:int, alpha:float):
    os.system(f"python3 reconstructor.py -ep={ep} -in=gansan -lr={lr} -bs={bs} -a={alpha} -n=\'grid-search\'")

parser = argparse.ArgumentParser()
args = parse_arguments(parser)

lrs = [1e-7]
bses = [2048, 4096]
input_dataset = args.input_dataset

Parallel(n_jobs=args.cpu_parallel)(delayed(launch)(bs, lr, args.epochs, args.alpha) for lr in lrs for bs in bses)

# Send Notification that Job is completed
text = f"Grid Search on {input_dataset} with {args.epochs} epochs, learning rates = {lrs} and batch sizes = {bses} Completed."
user_host = os.getenv("USER") + "@" + os.uname()[1]
content = 'Subject: %s\n\n%s' % (f"Python Job Completed on {user_host}", text)

mail = smtplib.SMTP('smtp.gmail.com',587)
mail.ehlo()
mail.starttls()
passw = os.getenv("python_pass")
mail.login('jc.taillandier1@gmail.com', passw)
mail.sendmail('jc.taillandier1@gmail.com','jc.taillandier@hotmail.com',content) 
mail.close()
