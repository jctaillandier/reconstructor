

# Personal Imports
from Modules import analysisTools as at
from Modules import utilities as utils
from Modules import datasets as d
from Modules import results as r

# from Stats import Plots as Pl
import csv
import sys
import yaml
import math
import json
import tqdm
import time
import torch
import random
import os.path
import warnings
import argparse
import importlib
import torchvision
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from typing import List
from torch.utils.data import *
import matplotlib.pyplot as plt
from torch.utils import data as td
from torch.autograd import Variable
from joblib import Parallel, delayed


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
GET_VALUE = lambda x: x.to(CPU_DEVICE).data.numpy().reshape(-1)[0]
print(f"\n *** \n Currently running on {device}\n *** \n")


# parser = argparse.ArgumentParser()
# args = utils. parse_arguments(parser)
a = 'adult'
if not sys.argv:
    raise EnvironmentError("Not enough parameters added to command: Need (1) paramsfile.yaml and (2) experiment_name")
args = sys.argv[1:]

if args[0][-5:] != '.yaml':
    raise ValueError("first argument needs to be the parameters file in yaml format")
params_file = args[0]

if args[1]:
    exp_name = args[1]
else:
    print("No name given to experiment.")
    exp_name = "no-name"

if len(args)>2 and args[2] == 'False':
    clean_data = False
else:
    clean_data = True

path_to_exp = utils.check_dir_path(f'./experiments/{exp_name}/')
os.mkdir(path_to_exp)
model_saved = path_to_exp+'models_data/'
os.mkdir(model_saved)
# Data import & Pre-processing

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

        # TODO Unsustainable implementation to remove '?' rows -> 30 sec for 48 000 X 15 np.array
        #   Only issue is when calculating diversity, cant convert ? to int or float
        # Save in CSV and use that as source is one solution
        if clean_data == True:
            df2 = utils.rm_qmark(df2)
            self.df_label = utils.rm_qmark(df_labels)

        # Useinput percentage for size of train / test split
        self.n_test = int(len(df2.iloc[:,0])*percent_train_set)

        #Make encoding object to transform categorical
        self.data_pp = d.Encoder(import_path)
        self.data_pp.fit_transform()
        self.data_pp.save_parameters(path_to_exp, prmFile="parameters_data.prm")

        # For Label (orginal_images)
        self.labels_pp = d.Encoder(import_path)
        self.labels_pp.fit_transform()
        self.labels_pp.save_parameters(path_to_exp, prmFile="parameters_labels.prm")
        
        

        print(f"Saving output dataset under ./data/*_NoCat.csv \n")
        self.data_pp.df.to_csv(f"{import_path[:-4]}_NoCat.csv", index=False)
        self.labels_pp.df.to_csv(f"{label_path[:-4]}_NoCat.csv", index=False)

        

        # MAke sure post processing label and data are of same shape
        if self.data_pp.df.shape != self.data_pp.df.shape:
            raise ValueError(f"The data csv ({self.data_pp.df.shape}) and labels ({self.data_pp.df.shape}) post-encoding don't have the same shape.")

        self.dataloader = My_dataLoader(self.batchSize, f"{import_path[:-4]}_NoCat.csv", f"{label_path[:-4]}_NoCat.csv", self.n_test, test_batch_size)

        self.data_dataframe = self.data_pp.df
        self.labels_dataframe = self.labels_pp.df

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

def test(model: torch.nn.Module, experiment: PreProcessing, test_loss_fn:torch.optim) -> (int, pd.DataFrame, pd.DataFrame):
    '''
        Does the test loop and if last epoch, decodes data, generates new data, returns and saves both 
        under ./experiments/<experiment_name>/

        :PARAMS
        model: torch model to use
        experiment: PreProcessing object that contains data
        test_loss_fn: Loss function from torch.nn 
    '''
    model.eval()
    
    test_loss = 0
    test_size = 0
    batch_ave = 0
    with torch.no_grad():
        for inputs, target in experiment.dataloader.test_loader:

            inputs, target = inputs.to(device), target.to(device)
        
            np_inputs = inputs.numpy()
            headers = experiment.data_pp.encoded_features_order
            input_df = pd.DataFrame(np_inputs, columns=headers)
            san_data = experiment.data_pp.__from_dummies__(ext_data=input_df)

            output = model(inputs.float())

            # decode dummy vars from model-generated data 
            #       borrowing inv_transform function, also need to add headers
            np_output = output.numpy()
            headers = experiment.data_pp.encoded_features_order
            output_df = pd.DataFrame(np_output, columns=headers)
            gen_data = experiment.data_pp.__from_dummies__(ext_data=output_df)

            test_size = len(inputs.float())
            test_loss += test_loss_fn(output.float(), target.float()).item() 
            batch_ave += test_loss/test_size
    
    return batch_ave, san_data, gen_data 


class Training:
    def __init__(self, experiment_x: PreProcessing, model_type:str='autoencoder'):
        self.wd = experiment_x.params['model_params']['weight_decay']['value']
        self.learning_rate = experiment_x.params['model_params']['learning_rate']['value']
        self.experiment_x = experiment_x
        self.num_epochs = int(experiment_x.params['model_params']['num_epochs']['value'])
        self.dim_red = experiment_x.params['post_processing']['dim_reduction']['value']

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
            Generates data every epoch but only saves on file if loss is lowest so far.
            
        
        '''
        start = time.time()
        lowest_test_loss = 9999999999999
        self.test_accuracy = []
        self.ave_train_loss = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch} of {self.num_epochs} running...")

            # Iterate on train set with SGD (adam)
            batch_ave_tr_loss = train(self.model,self.experiment_x.dataloader.train_loader, self.optimizer, self.train_loss)
            self.ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())

            # Check test set metrics (+ generate data if last epoch )
            loss, san_data, model_gen_data = test(self.model, self.experiment_x, self.test_loss_fn)

            if loss < lowest_test_loss:
                lowest_test_loss = loss
                fm = open(model_saved+f"lowest-test-loss_ep{epoch}.pth", "wb")
                torch.save(self.model.state_dict(), fm)
                fm.close()

                # Then we want this to be used as generated data
                model_gen_data.to_csv(f"{model_saved}lowest_loss_ep{epoch}.csv", index=False)
                self.san_data = san_data
                self.model_gen_data = model_gen_data


            print(f"Epoch {epoch} complete. Test Loss: {loss:.6f} \n")      
            self.test_accuracy.append(loss)#/len(self.experiment_x.dataloader.test_loader.dataset))

        if lowest_test_loss == loss: # redundant statement
            self.san_data = san_data
            self.model_gen_data = model_gen_data

        fm = open(model_saved+f"final-model_{self.num_epochs}ep.pth", "wb")
        torch.save(self.model.state_dict(), fm)

        end = time.time()
        print(f"Training on {self.num_epochs} epochs completed in {(end-start)/60} minutes.\n")

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
        print(f"Starting calculation Three-way of Damage (with {self.dim_red}) and Diversity...")
        start = time.time()
        
        pd_og_data = self.experiment_x.labels_pp.__from_dummies__(ext_data=self.experiment_x.labels_pp.df).iloc[self.experiment_x.dataloader.train_size:,:]
        pd_og_data.reset_index(drop=True,inplace=True)
        pd_san_data = self.san_data
        pd_gen_data = self.model_gen_data
        
        # Damage First / attr
        dam = at.Damage()
        dam_dict = {}

        test_data_dict = {}
        test_data_dict['orig_san'] = [pd_og_data, pd_san_data]
        test_data_dict['san_gen'] = [pd_san_data, pd_gen_data]
        test_data_dict['gen_orig'] = [pd_gen_data, pd_og_data] 
       

        # Visualisation for all three saved under <experiment_name>/dim_reduce/
        diversities = []
        os.mkdir(f"{path_to_exp}dim_reduce/")
        for key in test_data_dict:
            
            d_cat, d_num = dam(original=test_data_dict[key][0], transformed=test_data_dict[key][1])
            dam_dict[key] = [d_cat, d_num]

            if self.dim_red != 'none':
                dr = at.DimensionalityReduction()
                dr.clusters_original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]},labels=pd_og_data.iloc[:,5], dimRedFn=self.dim_red, savefig=path_to_exp+f"dim_reduce/{key}_damage.png")

                dr.original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]}, dimRedFn=self.dim_red, savefig=path_to_exp+f"dim_reduce/{key}_damage.png")

                # Then Diversity 
                # div = at.Diversity()
                # diversity = div(test_data_dict[key][0], f"original_{key.split('_')[0]}")
                # diversity.update(div(test_data_dict[key][1], f"transformed_{key.split('_')[1]}"))
                # diversities.append(diversity)
                diversity = []
                saver = r.DiversityDamageResults(resultDir=path_to_exp+f"dim_reduce/", result_file=f"cat_damage{key}",  num_damage=f"numerical_damage_{key}", overwrite=True)

                saver.add_results(diversity=diversity, damage_categorical=d_cat, damage_numerical=d_num, epoch=self.num_epochs, alpha_=0)
                
        end = time.time()
        # Save all in file TODO graphs
        with open(path_to_exp+f"dim_reduce/dam_div_metadata.txt", 'w+') as f:
            f.write(f"DAMAGE:\n")
            f.write(f"Damage from original to sanitized: {dam_dict['orig_san']}\n")
            f.write(f"Damage from sanitized to generated: {dam_dict['san_gen']}\n")
            f.write(f"Damage from generated to original: {dam_dict['gen_orig']}\n")
            f.write(f"Diversity:\n {diversities}")
            f.write(f"Time to generate graphs: {(end-start)/60:.2f} minutes")
        print(f"Calculations for Diversity and Damage compleded.")
        print(f"Results can be found under {path_to_exp}dim_reduce/")

    def gen_loss_graphs(self):
        
        a = f'adult_{self.num_epochs}ep'
        x_axis = np.arange(1,self.num_epochs+1)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(x_axis, self.test_accuracy)
        plt.xlabel("Epochs")
        plt.ylabel("L1 Loss")
        plt.title("Test Loss")

        plt.subplot(1,2,2)
        plt.plot(x_axis, self.ave_train_loss)
        plt.xlabel("Epochs")
        plt.ylabel("L1 Loss")
        plt.title("Train Loss")
        plt.savefig(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}_train-loss.png")

    




        
if __name__ == '__main__':

    # Pre Process Data
    experiment = PreProcessing(params_file)

    # Create training instance
    training_instance = Training(experiment)

    # Train the AE
    training_instance.train_model()

    # Run Metrics Calculation
    training_instance.post_training_metrics()

    # Generate test and train loss graphs (L1)
    training_instance.gen_loss_graphs()

    print(f"Experiment can be found under {path_to_exp}")

