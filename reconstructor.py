

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
    def __init__(self, batch_size : int, data_path :str, label_path:str, n_train :int,  label_col_name:str, test_batch_size:int=128):
        '''
            Creates train and test loaders from local files, to be easily used by torch.nn
            
            :batch_size: int for size of training batches
            :data_path: path to csv where data is. 2d file
            :label_path: csv containing labels. line by line equivalent to data_path file
            :n_train: int for the size of training set (assigned randomly)
            :test_batch: size of batches at test time. If none, will be same 
                    as training
            :label_col_name name of columns that contains labels
        '''
        self.batch_size = batch_size
        self.train_size = n_train
        
        df_data = pd.read_csv(data_path)
        df_label = pd.read_csv(label_path)
        
        a = df_data.values
        b = df_label.values
        self.trdata = torch.tensor(a[:self.train_size,:]) # where data is 2d [D_train_size x features]
        self.trlabels = torch.tensor(b[:self.train_size,:]) # also has too be 2d
        self.tedata = torch.tensor(a[self.train_size:,:]) # where data is 2d [D_train_size x features]
        self.telabels = torch.tensor(b[self.train_size:,:]) # also has too be 2d
        
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
        df_labels = pd.read_csv(label_path)
        n_test = int(len(df2.iloc[:,0])*percent_train_set)

        if df2.shape != df_labels.shape:
            print(df2.shape)
            print(df_labels.shape)
            raise ValueError("The labels csv and data csv don't have the same shape.")
        
        data_cat_list = utils.find_cat(df2)
        label_cat_list = utils.find_cat(df_labels)
        
        if data_cat_list != label_cat_list:
            raise ValueError('The categorical data columns found in your data doesnt match the ones found in your labels.\n That means the AE will learn on data that is encoded differently from the labels it tries to immitate.') 
       
        
        if len(data_cat_list) > 0:
            print(f"Categorical variable found. Running Pandas dummy_variable encoding on {len(data_cat_list)} columns \n")
            df_data = utils.dummy_encode(df2, data_cat_list)
            df_labels = utils.dummy_encode(df_labels, label_cat_list)
            print(f"Saving encoded dataset under {import_path[:-4]}_NoCat.csv \n")
            df_data.to_csv(f"{import_path[:-4]}_NoCat.csv", index=False)
            
            df_data, df_labels = utils.adjust_enc_errors(df_data, df_labels)
            if df_data.shape != df_labels.shape:
                print(df_data.shape)
                print(df_labels.shape)
                raise ValueError("The labels csv and data post-encoding don't have the same shape.")
                
            df_labels.to_csv(f"{label_path[:-4]}_NoCat.csv", index=False)
            self.dataloader = My_dataLoader(self.batchSize, f"{import_path[:-4]}_NoCat.csv", f"{label_path[:-4]}_NoCat.csv", n_test, "income", test_batch_size)
            self.data_dataframe = df_data

        else: # no categorical vars found
            self.dataloader = My_dataLoader(self.batchSize, import_path, n_test, "income", test_batch_size)
            self.data_dataframe = df_data


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

def test(model, test_loader, test_loss_fn, last_epoch=False):
    model.eval()
    
    test_loss = 0
    test_size = 0
    batch_ave = 0
    with torch.no_grad():
        for inputs, target in test_loader:

            inputs, target = inputs.to(device), target.to(device)
            
            if last_epoch == True:
                og_data = inputs.tolist()
                with open(path_to_exp+f"{str.replace(time.ctime()[4:-8], ' ', '_')}-original_testset.csv", 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(experiment.data_dataframe.columns.values)
                    writer.writerows(og_data)
            else:
                og_data = []

            output = model(inputs.float())

            if last_epoch == True:
                gen_data = output.tolist()
                with open(path_to_exp+f"{str.replace(time.ctime()[4:-8], ' ', '_')}-generated_testset.csv", 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(experiment.data_dataframe.columns.values)
                    writer.writerows(gen_data)
            else:
                gen_data = []

            test_size = len(inputs.float())
            test_loss += test_loss_fn(output.float(), target.float()).item() 
            batch_ave += test_loss/test_size
    
    return batch_ave, og_data, gen_data


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

            batch_ave_tr_loss = train(self.model,self.experiment_x.dataloader.train_loader, self.optimizer, self.train_loss)
            self.ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())
            last = False
            if epoch+1 == self.num_epochs: # then save the test batch input and output for metrics
                last = True
            loss, san_data, gen_data = test(self.model, self.experiment_x.dataloader.test_loader, self.test_loss_fn, last)

            print(f"Epoch {epoch} complete. Test Loss: {loss:.4f} \n")      
            self.test_accuracy.append(loss/len(self.experiment_x.dataloader.test_loader.dataset))
        if last:
            self.san_data = san_data
            self.gen_data = gen_data


        a = f'adult_{self.num_epochs}ep'

        fm = open(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}.pth", "wb")
        torch.save(self.model.state_dict(), fm)

        end = time.time()
        print(f"Training on {self.num_epochs} epochs completed in {(end-start)/60} minutes.")

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
            This will calculate diversity within generated dataset 
                And the damage the generated dataset has
                    Those will be compared to both the original and sanitized

            
        '''

        pd_og_data = pd.DataFrame(self.experiment_x.dataloader.telabels.numpy())
        pd_san_data = pd.DataFrame(self.san_data)
        pd_gen_data = pd.DataFrame(self.gen_data)

        # Damage First
        dam = at.Damage()
        # Original <-> Sanitized
        os_d_cat, os_d_num = dam(original=pd_og_data, transformed=pd_san_data)
        # Sanitized <-> generated
        sg_d_cat, sg_d_num =dam(original=pd_san_data, transformed=pd_gen_data)
        # Generated <-> Original
        go_d_cat, go_d_num =dam(original=pd_gen_data, transformed=pd_og_data)

        # Visualisation:
        dr = at.DimensionalityReduction()
        dr.clusters_original_vs_transformed_plots({"original": pd_og_data, "transformed": pd_san_data},
                                                  labels=pd_og_data[20],
                                                  savefig=path_to_exp+"damage.png")
        dr.original_vs_transformed_plots({"original": pd_og_data, "transformed": pd_san_data}, savefig=path_to_exp+"damage2.png")
        
        # pdb.set_trace()

        # Saving all results:


        # Then Diversity 

        
        # Save all in file TODO graphs
        with open(path_to_exp+f"_dam_div.txt", 'w+') as f:
            f.write(f"")

if __name__ == '__main__':
    experiment = PreProcessing(params_file)
    # pdb.set_trace()
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

