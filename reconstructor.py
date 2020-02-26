

# # Imports
from Modules import utilities as utils
from Modules import analysisTools as at
from Modules import datasets as d
# import datasets as d

# from Stats import Plots as Pl
import tqdm
import time
import math
import csv
import pdb
import sys
import yaml
import json
import torch
import argparse
import random
import os.path
import warnings
import importlib
import torchvision
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from typing import List
from torch.utils.data import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils import data as td
from joblib import Parallel, delayed


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
GET_VALUE = lambda x: x.to(CPU_DEVICE).data.numpy().reshape(-1)[0]
print(f"\n *** \n Currently running on {device}\n *** \n")


# parser = argparse.ArgumentParser()
# args = utils. parse_arguments(parser)

if not sys.argv:
    raise EnvironmentError("Not enough parameters added to command: Need (1) paramsfile.yaml and (2) experiment_name")
args = sys.argv[1:]

if args[0][-5:] != '.yaml':
    raise ValueError("first argument needs to be the parmeters file in yaml format")
params_file = args[0]

if args[1]:
    exp_name = args[1]
else:
    print("No name given to experiment.")
    exp_name = "no-name"

path_to_exp = utils.check_dir_path(f'./experiments/{exp_name}/')
os.mkdir(path_to_exp)


# Data import & Pre-processing
  
class My_dataLoader:
    def __init__(self, batch_size : int, data_path :str, label_path:str, n_train :int, test_batch_size:int=128):
        '''
            Creates train and test loaders from local files, to be easily used by torch.nn
            
            :batch_size: int for size of training batches
            :data_path: path to csv where data is. 2d file
            :label_path: csv containing labels. line by line equivalent to data_path file
            :n_train: int for the size of training set (assigned randomly)
            :test_batch_size: size of batches at test time. If none, will be same 
                    as training
        '''
        self.batch_size = batch_size
        self.train_size = n_train
        
        df_data = pd.read_csv(data_path)
        df_label = pd.read_csv(label_path)
        
        a = df_data.values
        self.b = df_label.values
        self.trdata = torch.tensor(a[:self.train_size,:]) # where data is 2d [D_train_size x features]
        self.trlabels = torch.tensor(self.b[:self.train_size,:]) # also has too be 2d
        self.tedata = torch.tensor(a[self.train_size:,:]) # where data is 2d [D_train_size x features]
        self.telabels = torch.tensor(self.b[self.train_size:,:]) # also has too be 2d
        
        self.train_dataset = torch.utils.data.TensorDataset(self.trdata, self.trlabels)
        self.test_dataset = torch.utils.data.TensorDataset(self.tedata, self.telabels)

        # Split dataset into train and Test sets
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=False
        )
        if test_batch_size == 'full':
            test_batch_size = len(self.test_dataset)
            
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            num_workers=1,
            pin_memory=False
        )

class PreProcessing:
    def __init__(self, params_file: str):
        '''
            Imports all variables/parameters necessary for preparation of training.
            Looks into params.yaml, then creates dataloader that can be used
            
            params_file: path to file that contains parameters.
                            Needs to follow a specific naming and format
        '''
        # Import params
        stream = open(params_file, 'r')
        self.params = yaml.load(stream, yaml.FullLoader)

        import_path = self.params['data_loading']['data_path']['value']
        label_path = self.params['data_loading']['label_path']['value']

        self.batchSize = self.params['model_params']['batchSize']['value']        
        test_batch_size = self.params['model_params']['test_batch_size']['value']        
        percent_train_set = self.params['model_params']['percent_train_set']['value']


        df2 = pd.read_csv(import_path)
        self.df_labels = pd.read_csv(label_path)
        n_test = int(len(df2.iloc[:,0])*percent_train_set)

        #Make encoding object to transform categorical
        self.data_pp = utils.encode(import_path)
        self.labels_pp = utils.encode(label_path)
        print(f"Saving output dataset under {import_path[:-4]}_NoCat.csv \n")
        self.data_pp.df.to_csv(f"{import_path[:-4]}_NoCat.csv", index=False)
        self.labels_pp.df.to_csv(f"{label_path[:-4]}_NoCat.csv", index=False)

        # MAke sure post processing label and data are of same shape
        if self.data_pp.df.shape != self.data_pp.df.shape:
            raise ValueError(f"The data csv ({self.data_pp.df.shape}) and labels ({self.data_pp.df.shape}) post-encoding don't have the same shape.")

        self.dataloader = My_dataLoader(self.batchSize, f"{import_path[:-4]}_NoCat.csv", f"{label_path[:-4]}_NoCat.csv", n_test, test_batch_size)

        self.data_dataframe = self.data_pp.df
        self.labels_dataframe = self.labels_pp.df


class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            # batchnorm?,
            nn.LeakyReLU(),
            nn.Linear(in_dim,in_dim),
            # batchnorm,
            nn.LeakyReLU(), 
            nn.Linear(in_dim,out_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            # batchnorm,
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(), 
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.decoder(x)
        
        return x
    


def train(model: torch.nn.Module, train_loader:torch.utils.data.DataLoader, optimizer:torch.optim, loss_fn) -> int:
    model.train()
    train_loss = []
    for batch_idx, (inputs, target) in enumerate(train_loader):
      
        inputs, target = inputs.to(device), target.to(device)
        
        
        output = model(inputs.float())
        loss_vector = loss_fn(output.float(), target.float())
        
        train_loss.append(sum(loss_vector))
    
        loss_per_dim = torch.sum(loss_vector, dim=0) 
        
        for loss in loss_per_dim:
            loss.backward(retain_graph=True)
            optimizer.step()
    mean_loss = sum(train_loss) / batch_idx+1
    mean_loss = mean_loss.detach()
    return sum(mean_loss)

def test(model: torch.nn.Module, experiment: PreProcessing, test_loss_fn:torch.optim, last_epoch: bool = False) -> (int, pd.DataFrame, pd.DataFrame):
    '''
        Does the test loop and if last epoch, decodes data, generates new data, returns and saves both 
        under ./experiments/<experiment_name>/

        :PARAMS
        model: torch model to use
    '''
    model.eval()
    
    test_loss = 0
    test_size = 0
    batch_ave = 0
    with torch.no_grad():
        for inputs, target in experiment.dataloader.test_loader:

            inputs, target = inputs.to(device), target.to(device)
            
            if last_epoch == True:

                np_inputs = inputs.numpy()
                headers = experiment.data_pp.encoded_features_order
                input_df = pd.DataFrame(np_inputs, columns=headers)
                san_data = experiment.data_pp.__from_dummies__(ext_data=input_df)
            else:
                san_data = []

            output = model(inputs.float())

            if last_epoch == True:
                # decode dummy vars from model-generated data 
                #       borrowing inv_transform function, also need to add headers
                np_output = output.numpy()
                headers = experiment.data_pp.encoded_features_order
                output_df = pd.DataFrame(np_output, columns=headers)
                gen_data = experiment.data_pp.__from_dummies__(ext_data=output_df)

            else:
                gen_data = []

            test_size = len(inputs.float())
            test_loss += test_loss_fn(output.float(), target.float()).item() 
            batch_ave += test_loss/test_size
    
    return batch_ave, san_data, gen_data # this needs to be decoded, but we cant directyl with d.de


class Training:
    def __init__(self, experiment_x: PreProcessing, model_type:str='autoencoder'):
        self.wd = experiment_x.params['model_params']['weight_decay']['value']
        self.learning_rate = experiment_x.params['model_params']['learning_rate']['value']
        self.experiment_x = experiment_x
        self.num_epochs = int(experiment_x.params['model_params']['num_epochs']['value'])

        self.in_dim = experiment_x.dataloader.trdata.shape[1]
        self.out_dim = experiment_x.dataloader.trlabels.shape[1]

        if model_type == 'autoencoder':
            self.model = Autoencoder(self.in_dim, self.out_dim).to(device)
        self.train_loss = torch.nn.L1Loss(reduction='none').to(device) 
        self.test_loss_fn =torch.nn.L1Loss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.wd)

    def train_model(self):
        '''
            Takes care of all model training, testing and generation of data
            
            experiment_x: PreProcessing object that contains all data and parameters
        
        '''
        start = time.time()

        self.test_accuracy = []
        self.ave_train_loss = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1} of {self.num_epochs} running...")

            # Iterate on train set with SGD (adam)
            batch_ave_tr_loss = train(self.model,self.experiment_x.dataloader.train_loader, self.optimizer, self.train_loss)
            self.ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())

            # check if last epoch, in order to generate data
            last = False
            if epoch+1 == self.num_epochs: 
                last = True

            # Check test set metrics (+ generate data if last epoch )
            loss, san_data, gen_data = test(self.model, self.experiment_x, self.test_loss_fn, last)

            print(f"Epoch {epoch+1} complete. Test Loss: {loss:.6f} \n")      
            self.test_accuracy.append(loss)#/len(self.experiment_x.dataloader.test_loader.dataset))
        if last: # redundant statement
            self.san_data = san_data
            self.gen_data = gen_data


        a = f'adult_{self.num_epochs}ep'

        fm = open(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}.pth", "wb")
        torch.save(self.model.state_dict(), fm)

        end = time.time()
        print(f"Training on {self.num_epochs} epochs completed in {(end-start)/60} minutes.")

        # Save model meta data in txt file
        with open(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}.txt", 'w+') as f:
            f.write(f"Epochs: {self.num_epochs} \n")
            f.write(f"Learning Rate: {self.learning_rate} \n")
            f.write(f"weight decay: {self.wd}\n")
            f.write(f"Training Loss: {str(self.train_loss)}\n")
            f.write(f"Test Loss: {str(self.test_loss_fn)} \n")
            f.write(f"self.self.optimizer: {str(self.optimizer)}\n")
            f.write(f"Model Architecture: {self.model}\n")
            f.write(f"Training completed in: {(end-start)/60:.2f} minutes\n")
            f.write(f"Train loss values: {str(self.ave_train_loss)} \n")
            f.write(f"Test loss values: {str(self.test_accuracy)}\n")


    def post_training_metrics(self):
        '''
            This will calculate (1) diversity within generated dataset, 
                and the (2) damage the generated dataset has
                    Those will be compared to both the original and sanitized            
        '''
        # Need to created a Encoder object with original data justin order to have matching columns when calculating damage and Diversity
        start = time.time()
        # a = pd.DataFrame(self.experiment_x.dataloader.telabels.numpy(), columns=self.experiment_x.labels_dataframe.columns)
        pd_og_data = self.experiment_x.labels_pp.__from_dummies__(ext_data=self.experiment_x.labels_pp.df).iloc[self.experiment_x.dataloader.train_size:,:]
        pd_og_data.reset_index(drop=True,inplace=True)
        pd_san_data = self.san_data
        pd_gen_data = self.gen_data
        
        # Damage First / attr
        dam = at.Damage()
        dam_dict = {}

        test_data_dict = {}
        test_data_dict['orig_san'] = [pd_og_data, pd_san_data]
        test_data_dict['san_gen'] = [pd_san_data, pd_gen_data]
        test_data_dict['gen_orig'] = [pd_gen_data, pd_og_data] 

        
        # Original <-> Sanitized
        os_d_cat, os_d_num = dam(original=pd_og_data, transformed=pd_san_data)
        dam_dict['original_sanitized'] = [os_d_cat, os_d_num] # output is dict per columns or if numerical, returns full dataset, with data on each deviation line by line
        # Sanitized <-> generated
        sg_d_cat, sg_d_num = dam(original=pd_san_data, transformed=pd_gen_data)
        dam_dict['sanitized_generated'] = [sg_d_cat, sg_d_num]
        # Generated <-> Original
        go_d_cat, go_d_num = dam(original=pd_gen_data, transformed=pd_og_data)
        dam_dict['generated_original'] = [go_d_cat, go_d_num]        

        # Visualisation for all three saved under <experiment_name>/dim_reduce/
        os.mkdir(f"{path_to_exp}dim_reduce/")
        for key in test_data_dict:
            dr = at.DimensionalityReduction()
            dr.clusters_original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]},
                                                    labels=pd_og_data.iloc[:,5], dimRedFn='umap',
                                                    savefig=path_to_exp+f"dim_reduce/{key}_damage.png")
            dr.original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]}, dimRedFn='umap',
                                                savefig=path_to_exp+f"dim_reduce/{key}_damage.png")

        # Then Diversity 
        for key in test_data_dict:
            div = at.Diversity()
            diversity = div(test_data_dict[key][0], f"original_{key.split('_')[0]}")
            diversity.update(div(test_data_dict[key][1], f"transformed_{key.split('_')[1]}"))

        end = time.time()

        # Save all in file TODO graphs
        with open(path_to_exp+f"dim_reduce/dam_div_metadata.txt", 'w+') as f:
            f.write(f"DAMAGE:\n")
            f.write(f"Damage from original to sanitized: {dam_dict['orig_san']}\n")
            f.write(f"Damage from sanitized to generated: {dam_dict['san_gen']}\n")
            f.write(f"Damage from generated to original: {dam_dict['gen_orig']}\n")
            f.write(f"Time to generate graphs: {(end-start)/60:.2f} minutes")

if __name__ == '__main__':
    experiment = PreProcessing(params_file)

    my_training = Training(experiment)
    my_training.train_model()

    my_training.post_training_metrics()

    # TODO Abstract in class of function
    a = f'adult_{my_training.num_epochs}ep'
    x_axis = np.arange(1,my_training.num_epochs+1)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(x_axis, my_training.test_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("L1 Loss")
    plt.title("Test Loss")

    plt.subplot(1,2,2)
    plt.plot(x_axis, my_training.ave_train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("L1 Loss")
    plt.title("Train Loss")
    plt.savefig(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}_train-loss.png")

    

    print(f"Experiment can be found under {path_to_exp}")

