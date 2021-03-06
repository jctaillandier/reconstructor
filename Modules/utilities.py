from typing import List
from Modules import datasets as d
from Modules import analysisTools as at
import pandas as pd
import numpy as np
import os, argparse, torch
    

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
    parser.add_argument('-n','--exp_name', type=str, default='exp', help='Name of the experiment. This will be used as folder name under /experiments/ dir', required=False)
    parser.add_argument('-ep','--epochs', type=int, default=100,help='Number of epochs to train the model.', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=2048, help='batch size for Training loop. Test set will alwayas be the size of the test set (passed as one batch)', required=False)
    parser.add_argument('-tbs','--test_batch_size', type=str, default='full', help='Size of test batch size. Do not touch. If fails for out of memory, need code adjustment', required=False)
    parser.add_argument('-pts','--percent_train_set', type=float, default=0.85, help='Percentage of total Dataset that will be kept for training. Rest will be used for testing', required=False)
    parser.add_argument('-wd','--weight_decay', type=float, default=0, help='Value for L2 penalty known as Weight Decay. Has not shown any value in this use case', required=False)
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-6, help='Learning rate on which we will optimize with Adam. Better performance are show with lr < 1e-4', required=False)
    parser.add_argument('-dr','--dim_red', type=str, default='none', help='Dimension reduction. Choose from pca, tsne, umap, svp or none. Where pca is quicker to compute', required=False, choices=['none', 'umap', 'tsne', 'svp','pca'])
    parser.add_argument('-a','--alpha', type=float, default=0.2, help='Value of alpha used when  sanitizing the dataset we use as input.', required=False)
    parser.add_argument('-gen','--attr_to_gen', type=str, default='none', help='Attribute that we want to remove from input if present, and have model infer; out_dim=in_dim+2', required=False)
    parser.add_argument('-in','--input_dataset', type=str, default='disp_impact', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=False, choices=['gansan', 'disp_impact'])
    parser.add_argument('-ns','--no_sex', type=str, default=True, help='Whether `sex` column is taken into account in training the reconstructor. Should be no', required=False)
    parser.add_argument('-kf','--kfold', type=str, default='false', help='Whether classifiers used after training to predict sensitive should use kfold')
    parser.add_argument('-mt','--model_type', type=str, default='autoencoder', help='autoencoder or VAE')

    # parser.add_argument('--discriminator_size', type=tuple, default=(256, 128, 1), help='The dimension size of the discriminator. (default value: (256, 128, 1))')
    args = parser.parse_args()
    if_alpha = f"_{args.alpha}a" if args.input_dataset == 'gansan' else ""
    exp_name = f"{args.model_type}_{args.input_dataset}{if_alpha}_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_" +args.exp_name
    path_to_exp = check_dir_path(f'./experiments/{exp_name}/')
    os.mkdir(path_to_exp)
    model_saved = path_to_exp+'models_data/'
    os.mkdir(model_saved)

    return args, path_to_exp, model_saved


def filter_cols(df, substr):
    '''
        iterates over headers and returns idexes of columns which contains the thing 
        and the ones that dont

        :PARAMS
        df: dataframe
        substr: substring we need in headers
    '''
    has = []
    hasnot = []
    for idx, header in enumerate(df.columns):
        if substr in header:
            has.append(idx)
        else:
            hasnot.append(idx)

    return has, hasnot
