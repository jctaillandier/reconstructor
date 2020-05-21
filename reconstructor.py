# Personal Imports
from Modules import analysisTools as at
from Modules import utilities as utils
from Modules import datasets as d
from Modules import results as r
from Modules import customLosses as cl
from Modules import classifiers 
from Modules.models import VAE, Autoencoder

# from Stats import Plots as Pl
import csv, sys, math, json, tqdm, time, torch, random, os.path, warnings, argparse, importlib, torchvision, pdb
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
from typing import List
from torch.utils.data import *
import matplotlib.pyplot as plt
from torch.utils import data as td
from torch.autograd import Variable
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

local_device = os.getenv("local_torch_device")
device = torch.device(local_device if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
CPU_DEVICE = torch.device("cpu")
GET_VALUE = lambda x: x.to(CPU_DEVICE).data.numpy().reshape(-1)[0]
print(f"\n *** \n Currently running on {device}\n *** \n")

class PreProcessing:
    def __init__(self):
        '''
            Imports all variables/parameters necessary for preparation of training.
            Looks into params.yaml, then creates dataloader that can be used
            
            params_file: path to file that contains parameters.
                            Needs to follow a specific naming and format
        '''

        if args.input_dataset == 'disp_impact': 
            import_path = "./data/disp_impact_remover_1.0.csv"
            label_path = "./data/disp_impact_remover_og.csv"

        elif args.input_dataset == 'gansan':    
            """
                Load the sanitized data from gansna with different values of alpha
                alpha=0 means no protection, a=0.9875 is optimal, ie: top protection
            """
            no_sex_file = ""
            if args.no_sex == True:
                no_sex_file = "_no_sex"
            if args.alpha == 0.2:
                import_path = f"./data/adult_sanitized_0.2{no_sex_file}.csv"
                sex_labels = pd.read_csv(f"./data/adult_sanitized_0.2.csv")
            elif args.alpha == 0.8:
                import_path = f"./data/adult_sanitized_0.8{no_sex_file}.csv"
                sex_labels = pd.read_csv(f"./data/adult_sanitized_0.8.csv")
            elif args.alpha == 0.9875:
                import_path = f"./data/adult_sanitized_0.9875{no_sex_file}.csv"
                sex_labels = pd.read_csv(f"./data/adult_sanitized_0.9875.csv")
            label_path = "./data/gansan_original.csv"

        if_gansan = f" with alpha = {args.alpha}" if args.input_dataset == 'gansan' else ""
        print(f"\n Launching attack on {args.input_dataset} dataset{if_gansan}. \n")
        
        self.attr_to_gen = args.attr_to_gen

        self.batchSize = args.batch_size       
           
        # Useinput percentage for size of train / test split   
        percent_train_set = args.percent_train_set
        df2 = pd.read_csv(import_path)
        self.n_test = int(len(df2.iloc[:,0])*percent_train_set)


        # Encode Input data if needed
        self.data_pp = d.Encoder(import_path)
        self.labels_pp = d.Encoder(label_path)

        # remove income as it should not be available in sanitized nor reconstruction
        # self.labels_pp.df.drop('income', axis=1, inplace=True)
        # self.data_pp.df.drop('income',axis=1, inplace=True)
        
        # Encode if gansan input
        if args.input_dataset == 'gansan':
            # self.data_pp.load_parameters('./data/')
            # self.labels_pp.load_parameters('./data/')
            self.data_pp.fit_transform()
            self.data_pp.save_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
            self.labels_pp.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
            self.labels_pp.transform()

        
        # MAke sure post processing label and data are of same shape
        if self.data_pp.df.shape != self.labels_pp.df.shape:
            raise ValueError(f"The data csv ({self.data_pp.df.shape}) and labels ({self.labels_pp.df.shape}) post-encoding don't have the same shape.")

        # self.data_pp.df.to_csv("/home/jc/Desktop/0.9875gansanitized_encoded.csv")
        self.dataloader = My_dataLoader(self.batchSize, self.data_pp, self.labels_pp, self.n_test,  sex_labels)
        

class My_dataLoader:
    def __init__(self, batch_size : int, df_data: d.Encoder, df_label: d.Encoder, n_train :int, sex_labels):
        '''
            Creates train and test loaders from local files, to be easily used by torch.nn
            
            :batch_size: int for size of training batches
            :data_path: path to csv where data is. 2d file
            :label_path: csv containing labels. line by line equivalent to data_path file
            :n_train: int for the size of training set (assigned randomly)
            :sex_labels  labels associated to data that will be generated by reconstructor
        '''
        self.batch_size = batch_size
        self.train_size = n_train
        self.cols_with_sensitive = df_data.df.columns
        self.data_with_sensitive = df_data.df
        self.label_with_sensitive = df_label.df

        # Remove Sensitive attribute from input since we want to infer it 
        # Assumption is that sanitizer will not provide it 
        # REMOVED self.col_rm columns 
        self.df_data = df_data.df
        self.df_label = df_label.df
        self.col_rm = 0
        if args.attr_to_gen.lower() is not 'none':
            for col in self.df_data.columns:
                if args.attr_to_gen.lower() in col.lower():
                    self.df_data = self.df_data.drop([col], axis=1)
                    self.col_rm = self.col_rm + 1
        
                    # self.out_dim_to_add = self.out_dim_to_add + 1
        self.headers_wo_sensitive =  self.df_data.columns   
        self.trdata = torch.tensor(self.df_data.values[:self.train_size,:]) # where data is 2d [D_train_size x features]
        self.trlabels = torch.tensor(self.df_label.values[:self.train_size,:]) # also has too be 2d
        self.tedata = torch.tensor(self.df_data.values[self.train_size:,:]) # where data is 2d [D_train_size x features]
        self.telabels = torch.tensor(self.df_label.values[self.train_size:,:]) # also has too be 2d  
        self.train_dataset = torch.utils.data.TensorDataset(self.trdata, self.trlabels)
        self.test_dataset = torch.utils.data.TensorDataset(self.tedata, self.telabels)

        self.sex_labelss = sex_labels['sex'].iloc[self.train_size:,]
        self.sex_labelss.reset_index(drop=True, inplace=True)
        # Split dataset into train and Test sets
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=False
        )
        if args.test_batch_size == 'full':
            self.test_batch_size = len(self.test_dataset)
        else:
            self.test_batch_size = args.test_batch_size
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=1,
            pin_memory=False
        )

        cat_ix = self.df_data.select_dtypes(include=['object', 'bool']).columns
        num_ix = self.df_data.select_dtypes(include=['int64', 'float64']).columns
        self.c_ix = []
        self.n_ix = []
        self.num_idx, self.cat_idx = utils.filter_cols(self.df_data, '=')

    
def train(model: torch.nn.Module, preprocessing:PreProcessing, optimizer:torch.optim, loss_fn) -> int:
    model.train()
    train_loss = []
    for batch_idx, (inputs, target) in enumerate(preprocessing.dataloader.train_loader):
        inputs, target = inputs.to(device), target.to(device)        
        

        if args.model_type =='vae':
            output, mu, logvar = model(inputs.float())
            loss_vector = cl.vae_loss(inputs.float(), target.float(), mu, logvar)
        else:
            output = model(inputs.float())
            loss_vector = loss_fn(output.float(), target.float())
        loss_per_dim = torch.sum(loss_vector, dim=0) 
        train_loss.append(sum(loss_per_dim)/len(loss_per_dim))
        count = 0
        for loss in loss_per_dim:
            loss.backward(retain_graph=True)
            optimizer.step()
            count +=1
    mean_loss = sum(train_loss) / batch_idx+1
    mean_loss = mean_loss.detach()
    
    return mean_loss

def test(model: torch.nn.Module, experiment: PreProcessing, test_loss_fn:torch.optim, last_epoch: bool) -> (int, pd.DataFrame):
    '''
        Does the test loop and if last epoch, decodes data, generates new data, returns and saves both 
        under ./experiments/<experiment_name>/

        :PARAMS
        model: torch model to use
        experiment: PreProcessing object that contains data
        test_loss_fn: Loss function from torch.nn 

        :RETURN
        int: average loss on this epoch
        pd.DataFrame: generated data in a dataframe with encoded data
        last_epoch: current epoch
    '''
    model.eval()
    
    batch_ave = 0
    with torch.no_grad():
        for inputs, target in experiment.dataloader.test_loader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs.float())
            if args.model_type == 'vae':
                pdb.set_trace()
                output = output[0]
            else:
                np_output = output
            headers = experiment.data_pp.encoded_features_order
            gen_data = pd.DataFrame(np_output, columns=headers)
            # here I keep values for L1 distance on each dimensions
            loss = test_loss_fn(output.float(), target.float())
            loss_per_dim = torch.sum(loss, dim=0) 
    
    if last_epoch == True:
        # To save sanitized test set to compare with generated data line by line
        data = pd.DataFrame(inputs.cpu().numpy(), columns=experiment.data_pp.encoded_features_order)
        data.to_csv(f"{model_saved}sanitized_testset_raw.csv", index=False)
        some_enc = d.Encoder(f"{model_saved}sanitized_testset_raw.csv")
        some_enc.load_parameters(path_to_exp,f"{args.input_dataset}_parameters_data.prm")
        some_enc.inverse_transform()
        final_df = pd.concat([some_enc.df, experiment.dataloader.sex_labelss], axis=1)
        final_df.to_csv(f"{model_saved}sanitized_testset_clean.csv", index=False)
    loss_ = loss_per_dim
    return loss_.detach(), gen_data 

class Training:
    def __init__(self, experiment_x: PreProcessing, model_type:str='autoencoder'):
        '''
            Training class takes in the Preprocessing object containing all data and creates an instance of training that is loading all hyper parameters from the specified file.
            It will allow the training and testing loops, as well as the generation of all relevant metrics for comparison and analysis post-training.
            Note that test loss is kept on a dimension by dimension basis, in order to identify which feature are getting close to original data and which aren't

            :PARAMS
            experiment_x: PreProcessing object created beforehand.  
        '''
        self.wd = args.weight_decay
        self.learning_rate = args.learning_rate
        self.experiment_x = experiment_x
        self.num_epochs = args.epochs
        self.dim_red = args.dim_red

        self.in_dim = experiment_x.dataloader.trdata.shape[1]
        # out dim = in dim + columns we removed if attr_to_gen
        self.out_dim = experiment_x.dataloader.trlabels.shape[1]

        if model_type == 'autoencoder':
            self.model = Autoencoder(self.in_dim, self.out_dim).to(device)
        elif model_type== 'vae':
            self.model = VAE(self.in_dim, self.out_dim).to(device)

        self.loss_fn = torch.nn.L1Loss(reduction='none').to(device) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.wd)

    def train_model(self):
        '''
            Takes care of all model training, testing and generation of data
            Generates data every epoch but only saves on file if loss is lowest so far.
        '''
        start = time.time()
        lowest_test_loss = 9999999999999999
        self.test_accuracy = []
        self.ave_train_loss = []
        self.best_generated_data = []
        self.lowest_loss_ep = -1
        self.lowest_loss_per_dim = []
        last_ep = False

        for epoch in tqdm.tqdm(range(self.num_epochs), desc=f"lr={args.learning_rate}, bs={args.batch_size}->"):
            # print(f"Running Epoch {epoch+1} / {self.num_epochs} for lr={args.learning_rate}, bs={args.batch_size}")
            # Iterate on train set with SGD (adam)
            batch_ave_tr_loss = train(self.model,self.experiment_x, self.optimizer, self.loss_fn)
            
            self.ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())

            if epoch+1 == self.num_epochs:
                last_ep=True
            loss_per_dim_per_epoch, model_gen_data = test(self.model, self.experiment_x, self.loss_fn, last_ep)
            loss = sum(loss_per_dim_per_epoch)/len(loss_per_dim_per_epoch)
            
            # To save as lowest loss averaged over all dim, for loss graph,
            #   NOT for comparison for each dimensions
            if loss < lowest_test_loss:
                if os.path.isfile(model_saved+f"lowest-test-loss-model_ep{self.lowest_loss_ep}.pth"):
                    os.remove(model_saved+f"lowest-test-loss-model_ep{self.lowest_loss_ep}.pth")
                lowest_test_loss = loss
                fm = open(model_saved+f"lowest-test-loss-model_ep{epoch}.pth", "wb")
                torch.save(self.model.state_dict(), fm)
                fm.close()
                self.lowest_loss_per_dim = loss_per_dim_per_epoch.cpu().numpy().tolist()
                self.lowest_loss_ep = epoch
                # Then we want this to be used as generated data
                self.best_generated_data = model_gen_data

            loss_df = pd.DataFrame([self.lowest_loss_per_dim], columns=self.experiment_x.data_pp.encoded_features_order)
            self.test_accuracy.append(loss)#/len(self.experiment_x.dataloader.test_loader.dataset))

        self.test_gen = pd.DataFrame(self.best_generated_data.values, columns=self.experiment_x.data_pp.encoded_features_order)
        self.test_gen.to_csv(f"{model_saved}best_loss_raw_generated.csv", index=False)
        if args.input_dataset =='gansan':
                self.gen_encoder = d.Encoder(f"{model_saved}best_loss_raw_generated.csv")
                self.gen_encoder.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
                self.gen_encoder.inverse_transform()
                final_df = pd.concat([self.gen_encoder.df, self.experiment_x.dataloader.sex_labelss], axis=1)
                # final_df_clean = pd.concat([self.gen_encoder.df, self.experiment_x.dataloader.sex_labelss], axis=1)
                final_df.to_csv(f"{model_saved}best_loss_clean_generated.csv", index=False)


        fm = open(model_saved+f"final-model_{self.num_epochs}ep.pth", "wb")
        torch.save(self.model.state_dict(), fm)

        # To save last epoch generated data
        last_ep_data = pd.DataFrame(model_gen_data, columns=self.experiment_x.data_pp.encoded_features_order)
        last_ep_data.to_csv(f"{model_saved}last_ep_data_raw.csv", index=False)
        some_enc = d.Encoder(f"{model_saved}last_ep_data_raw.csv")
        some_enc.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
        some_enc.inverse_transform()
        final_df = pd.concat([some_enc.df, self.experiment_x.dataloader.sex_labelss], axis=1)
        # final_df_clean = pd.concat([self.gen_encoder.df, self.experiment_x.dataloader.sex_labelss], axis=1)
        final_df.to_csv(f"{model_saved}last_ep_data_clean.csv", index=False)


        end = time.time()
        print(f"Training on {self.num_epochs} epochs completed in {(end-start)/60} minutes.\n")

        # Calculate and save the L1 distance on each encoded feature on Sanitized and Original Data, to compare and see whether the training got the distance closer. 
        san_loss_fn = torch.nn.L1Loss(reduction='none')
        og_test_data = torch.tensor(self.experiment_x.dataloader.data_with_sensitive.values[self.experiment_x.dataloader.train_size:,:])
        og_test_labels = torch.tensor(self.experiment_x.dataloader.label_with_sensitive.values[self.experiment_x.dataloader.train_size:,:])
        self.og_test_dataset = torch.utils.data.TensorDataset(og_test_data, og_test_labels)
         
        og_dataloader = torch.utils.data.DataLoader(
            self.og_test_dataset,
            batch_size=self.experiment_x.dataloader.test_batch_size,
            num_workers=1,
            pin_memory=False
        )       
        # Loss between saniized and original data. This is what we want to be worse than our recons loss
        for inputs, target in og_dataloader:
            loss_dim_batch = san_loss_fn(inputs.float(), target.float())
            loss_dim = loss_dim_batch.mean(dim=0)
            self.sanitized_loss = loss_dim.cpu().numpy().tolist()
            break

        # here we compare the above with L1 from data generated by reconstructor
        #   contained in self.lowest_loss_per_dim
        is_better = []
        how_much = []
        dtype = []
        better_count = 0
        for i, loss in enumerate(self.sanitized_loss):
            how_much.append((loss - self.lowest_loss_per_dim[i]))
            if self.lowest_loss_per_dim[i] < loss:
                is_better.append(True)
                better_count = better_count+1
            else:
                is_better.append(False)
         
        if args.input_dataset == 'gansan':
            columns=self.experiment_x.data_pp.encoded_features_order
        else:
            columns=self.experiment_x.data_pp.cols_order
        is_better_df = pd.DataFrame([self.sanitized_loss, self.lowest_loss_per_dim, is_better,how_much], columns=columns)
        # Save model meta data in txt file
        with open(path_to_exp+f"metadata.txt", 'w+') as f:
            f.write(f"Epochs: {self.num_epochs} \n \n")
            f.write(f"Total of {better_count} / {len(is_better_df.columns)} are now closer to original data (L1 distance). \n \n")
            f.write(f"Epoch of lowest loss: {self.lowest_loss_ep} \n \n")
            f.write(f"Average lowest loss: {sum(self.lowest_loss_per_dim)/len(self.lowest_loss_per_dim)}{self.lowest_loss_ep} \n \n")
            f.write(f"Lowest lost Generated:\n {self.lowest_loss_per_dim} \n \n")
            f.write(f"Loss from sanitized data: \n{self.sanitized_loss} \n")
            f.write(f"\n \n Learning Rate: {self.learning_rate} \n")
            f.write(f"Number Epochs: {self.num_epochs} \n")
            f.write(f"weight decay: {self.wd}\n")
            f.write(f"Training Loss: {str(self.loss_fn)}\n")
            f.write(f"self.self.optimizer: {str(self.optimizer)}\n")
            f.write(f"Model Architecture: {self.model}\n")
            f.write(f"Training completed in: {(end-start)/60:.2f} minutes\n")

        os.mkdir(path_base)
        is_better_df.to_csv(path_base+f"reconstruction_appraisal.csv", index=False, float_format='%.6f')
        

        
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
        plt.ylabel("Damage")
        plt.title("Train Loss on categorical features")


        if_gansan = f"_{args.alpha}a" if args.input_dataset == 'gansan' else ""
        plt.savefig(path_to_exp+f"{args.exp_name}-{a}{if_gansan}_{args.learning_rate}lr_{args.batch_size}.png")
    
    def pandas_describe(self):
        '''
            Runs and save pandas.describe function on all three dataset:
            Orginal, Sanitized and Generated, then saves under /base_metrics
            Also saves Generated data with encoded columns and (if gansan) original columns
        '''

        test_san = self.experiment_x.dataloader.data_with_sensitive.values[self.experiment_x.dataloader.train_size:,:] 
        headers = self.experiment_x.dataloader.cols_with_sensitive
        self.san_data = pd.DataFrame(test_san, columns=headers)

        test_og = self.experiment_x.dataloader.df_label.values[self.experiment_x.dataloader.train_size:,:]
        self.og_data = pd.DataFrame(test_og, columns=headers)
        

        a = self.og_data.describe()
        b = self.san_data.describe()
        c = self.test_gen.describe()
        
        
        a.to_csv(path_base+"original.csv")
        b.to_csv(path_base+"sanitized.csv")
        c.to_csv(path_base+"generated.csv")

def main():

    # Pre Process Data
    experiment = PreProcessing()

    # Create training instance
    training_instance = Training(experiment, model_type=args.model_type)

    # Train the AE
    training_instance.train_model()

    # Generate test and train loss graphs (L1)
    training_instance.gen_loss_graphs()
    training_instance.pandas_describe()
    print(f"Training completed. Running external Classifiers...")

    # classifiers predicting sex variable
    ext_classif = classifiers.BaseClassifiers("best_loss_clean_generated",path_to_exp, args.kfold)
    ext_classif.runit()
    # classifiers predicting sex variable
    ext_classif = classifiers.BaseClassifiers("last_ep_data_clean",path_to_exp, args.kfold)
    ext_classif.runit()


    print(f"\n \n Experiment can be found under {path_to_exp} \n \n ")

# Setup folders and global variables
parser = argparse.ArgumentParser()
args, path_to_exp, model_saved = utils.parse_arguments(parser)
path_base = path_to_exp+'distance_comparison/'

# Launch
main()
