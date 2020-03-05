from typing import List
from Modules import datasets as d
from Modules import analysisTools as at
import pandas as pd
import numpy as np
import os, argparse, torch, argparse
    

def check_dir_path(path_to_check: str) -> str:
    '''
        Checks if provided path is currently a at current level directory.
        If it is, it appends a number to the end and checks again 
        until no directory with such name exists

        :PARAMS
        path_to_check: str The path to location to check

        return: str New path with which os.mkdir can be called
    '''
    new_path = path_to_check
    if os.path.isdir(path_to_check):
        print("Experiment with name: \'{}\' already exists. Appending int to folder name. ".format(path_to_check))
        if os.path.isdir(path_to_check):
            expand = 1
            while True:
                expand += 1
                new_path = path_to_check[:-1] + '_' + str(expand) + '/'
                if os.path.isdir(new_path):
                    continue
                else:
                    break
            print(f"Experiment path: {new_path} \n \n ")
    return new_path

def tensor_to_df(tensor: torch.Tensor, headers: List[str]) -> pd.DataFrame:
    '''
        Takes in a 2d or 3d tensor and its column list and returns datafram
        if 3d, we assume first dim (dim=0) is the batch size, hence its ignored


    :PARAMS
    tensor: tensor to convert
    headers: list of headers to assign in that order. Must be same size 
                as last dim of tensor parameter.

    '''
    if tensor.shape[-1] != len(headers):
        raise ValueError(f"Tensor's last dimension ({tensor.shape[-1]}) must match headers length ({len(headers)})")

    
    return pd.DataFrame (tensor.tolist(), columns=headers)


def parse_arguments(parser):
    parser.add_argument('-n','--exp_name', type=str, default='no-name', help='Name of the experiment. This will be used as folder name under /experiments/ dir', required=True)
    parser.add_argument('-ep','--epochs', type=int, help='Number of epochs to train the model.', required=True)
    parser.add_argument('-p','--params', type=str, default='params.yaml', help='.yaml file containing most parameters for model')
    parser.add_argument('-bs','--batch_size', type=int, default=512, help='batch size for Training loop. Test set will alwayas be the size of the test set (passed as one batch)')
    parser.add_argument('-tbs','--test_batch_size', type=str, default='full', help='Size of test batch size. Do not touch. If fails for out of memory, need code adjustment', required=False)
    parser.add_argument('-pts','--percent_train_set', type=float, default=0.85, help='Percentage of total Dataset that will be kept for training. Rest will be used for testing', required=False)
    parser.add_argument('-wd','--weight_decay', type=float, default=0, help='Value for L2 penalty known as Weight Decay. Has not shown any value in this use case', required=False)
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-5, help='Learning rate on which we will optimize with Adam. Better performance are show with lr < 1e-4', required=False)
    parser.add_argument('-dr','--dim_red', type=str, default='umap', help='Dimension reduction. Choose from pca, tsne, umap, svp or none. Where pca is quicker to compute', required=False)
    parser.add_argument('-a','--alpha', type=float, default=0, help='Value of alpha used when  sanitizing the dataset we use as input.', required=False)
    # parser.add_argument('-','--', type=, default='', help='', required=False)

    # parser.add_argument('--discriminator_size', type=tuple, default=(256, 128, 1), help='The dimension size of the discriminator. (default value: (256, 128, 1))')
    args = parser.parse_args()

    path_to_exp = check_dir_path(f'./experiments/{args.exp_name}/')
    os.mkdir(path_to_exp)
    model_saved = path_to_exp+'models_data/'
    os.mkdir(model_saved)

    return args, path_to_exp, model_saved
