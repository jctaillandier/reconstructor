# Personal Imports
from Modules import analysisTools as at
from Modules import utilities as utils
from Modules import datasets as d
from Modules import results as r

# from Stats import Plots as Pl
import csv, sys, math, json, tqdm, time, torch, random, os.path, warnings, argparse, importlib, torchvision 
import numpy as np
import pdb
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
            import_path = "./data/0a_no1_e20.csv"
            label_path = "./data/gansan_original.csv"
        print(f"\n Running on {args.input_dataset} dataset input. \n")
        
        self.attr_to_gen = args.attr_to_gen

        self.batchSize = args.batch_size       
           
        # Useinput percentage for size of train / test split   
        percent_train_set = args.percent_train_set
        df2 = pd.read_csv(import_path)
        self.n_test = int(len(df2.iloc[:,0])*percent_train_set)


        # Encode Input data if needed
        self.data_pp = d.Encoder(import_path)
        self.labels_pp = d.Encoder(label_path)
        
        # Remove ? --> Likely no need if using gansan' AdultNotNA.csv
        # for index, col in enumerate(self.data_pp.df.columns):
        #     self.data_pp.df = self.data_pp.df[~self.data_pp.df[str(col)].isin(['?'])]
        #     self.labels_pp.df = self.labels_pp.df.drop(index=index)
        # for index, col in enumerate(self.labels_pp.df.columns):
        #     self.labels_pp.df = self.labels_pp.df[~self.labels_pp.df[str(col)].isin(['?'])]
        #     self.data_pp.df = self.data_pp.df.drop(index=index)
        
        # Encode if gansan input
        if args.input_dataset == 'gansan':
            # self.data_pp.load_parameters('./data/')
            # self.labels_pp.load_parameters('./data/')
            self.data_pp.fit_transform()
            self.data_pp.save_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
            self.labels_pp.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
            self.labels_pp.transform()
        else:
            self.data_pp.save_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
            # self.labels_pp.df.drop(['income-per-year'], axis=1, inplace=True)
            # self.data_pp.df.drop(['income-per-year'],axis=1, inplace=True)
        
        # MAke sure post processing label and data are of same shape
        if self.data_pp.df.shape != self.labels_pp.df.shape:
            raise ValueError(f"The data csv ({self.data_pp.df.shape}) and labels ({self.labels_pp.df.shape}) post-encoding don't have the same shape.")

        
        self.dataloader = My_dataLoader(self.batchSize, self.data_pp.df, self.labels_pp.df, self.n_test)
        

class My_dataLoader:
    def __init__(self, batch_size : int, df_data: pd.DataFrame, df_label:pd.DataFrame, n_train :int):
        '''
            Creates train and test loaders from local files, to be easily used by torch.nn
            
            :batch_size: int for size of training batches
            :data_path: path to csv where data is. 2d file
            :label_path: csv containing labels. line by line equivalent to data_path file
            :n_train: int for the size of training set (assigned randomly)
        '''
        self.batch_size = batch_size
        self.train_size = n_train
        self.data_with_sensitive = df_data
        self.label_with_sensitive = df_label
        # Remove Sensitive attribute from input since we want to infer it 
        # Assumption is that sanitizer will not provide it 
        # self.out_dim_to_add = 0 
        self.df_data = df_data
        self.df_label = df_label
        if args.attr_to_gen.lower() is not 'none':
            for col in df_data.columns:
                if args.attr_to_gen.lower() in col.lower():
                    df_data.drop([col], axis=1, inplace=True)
        
                    # self.out_dim_to_add = self.out_dim_to_add + 1
        self.headers_wo_sensitive =  df_data.columns   
        self.trdata = torch.tensor(self.df_data.values[:self.train_size,:]) # where data is 2d [D_train_size x features]
        self.trlabels = torch.tensor(self.df_label.values[:self.train_size,:]) # also has too be 2d
        self.tedata = torch.tensor(self.df_data.values[self.train_size:,:]) # where data is 2d [D_train_size x features]
        self.telabels = torch.tensor(self.df_label.values[self.train_size:,:]) # also has too be 2d  
        self.train_dataset = torch.utils.data.TensorDataset(self.trdata, self.trlabels)
        self.test_dataset = torch.utils.data.TensorDataset(self.tedata, self.telabels)
        
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

class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            # batchnorm?,
            nn.LeakyReLU(),
            nn.Linear(in_dim,out_dim),
            # batchnorm,
            nn.LeakyReLU(), 
            nn.Linear(out_dim,out_dim),
            nn.LeakyReLU()
        )

    def forward(self, xin):
        x = self.encoder(xin)
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
    return sum(mean_loss)/len(mean_loss)

def test(model: torch.nn.Module, experiment: PreProcessing, test_loss_fn:torch.optim) -> (int, pd.DataFrame):
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
    '''
    model.eval()
    
    test_loss = []
    batch_ave = 0
    with torch.no_grad():
        for inputs, target in experiment.dataloader.test_loader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs.float())
            
            np_output = output.cpu().numpy()
            headers = experiment.data_pp.encoded_features_order
            gen_data = pd.DataFrame(np_output, columns=headers)

            # here I keep values for L1 distance on each dimensions
            loss = test_loss_fn(output.float(), target.float())
    return loss.mean(dim=0), gen_data 

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
        self.out_dim = experiment_x.dataloader.trlabels.shape[1]

        if model_type == 'autoencoder':
            self.model = Autoencoder(self.in_dim, self.out_dim).to(device)
        self.train_loss = torch.nn.L1Loss(reduction='none').to(device) 
        self.test_loss_fn =torch.nn.L1Loss(reduction='none').to(device)
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
        self.best_generated_data = []
        self.lowest_loss_ep = -1
        self.lowest_loss_per_dim = []
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch} of {self.num_epochs} running...")

            # Iterate on train set with SGD (adam)
            batch_ave_tr_loss = train(self.model,self.experiment_x.dataloader.train_loader, self.optimizer, self.train_loss)
            self.ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())

            # Check test set metrics (+ generate data if last epoch )
            loss_per_dim_per_epoch, model_gen_data = test(self.model, self.experiment_x, self.test_loss_fn)
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
                self.lowest_loss_per_dim = loss_per_dim_per_epoch.numpy().tolist()
                self.lowest_loss_ep = epoch
                # Then we want this to be used as generated data
                self.best_generated_data = model_gen_data

            loss_df = pd.DataFrame([self.lowest_loss_per_dim], columns=self.experiment_x.data_pp.encoded_features_order)
            print(f"Epoch {epoch} complete.\n Test Loss on epoch: {loss_per_dim_per_epoch} \n")   
            print(f"Average Loss: {sum(loss_per_dim_per_epoch)/len(loss_per_dim_per_epoch)} \n ")   
            self.test_accuracy.append(loss)#/len(self.experiment_x.dataloader.test_loader.dataset))

        fm = open(model_saved+f"final-model_{self.num_epochs}ep.pth", "wb")
        torch.save(self.model.state_dict(), fm)

        end = time.time()
        print(f"Training on {self.num_epochs} epochs completed in {(end-start)/60} minutes.\n")

        # Calculate and save the L1 distance on each encoded feature on Sanitized and Original Data, to compare and see whether the training got the distance closer. We include the sensitive attribute.
        san_loss_fn = torch.nn.L1Loss(reduction='none')
        if args.input_dataset != 'gansan':
            ###
            ###     To be checked once i get data back
            ###
            og_test_data = torch.tensor(self.experiment_x.dataloader.data_with_sensitive.values[self.experiment_x.dataloader.train_size:,:])
            og_test_labels = torch.tensor(self.experiment_x.dataloader.label_with_sensitive.values[self.experiment_x.dataloader.train_size:,:])
            self.og_test_dataset = torch.utils.data.TensorDataset(og_test_data, og_test_labels)

        else:
            og_test_data = torch.tensor(self.experiment_x.dataloader.data_with_sensitive.values[self.experiment_x.dataloader.train_size:,:])
            og_test_labels = torch.tensor(self.experiment_x.dataloader.label_with_sensitive.values[self.experiment_x.dataloader.train_size:,:])
            self.og_test_dataset = torch.utils.data.TensorDataset(og_test_data, og_test_labels)

        og_dataloader = torch.utils.data.DataLoader(
            self.og_test_dataset,
            batch_size=self.experiment_x.dataloader.test_batch_size,
            num_workers=1,
            pin_memory=False
        )       
        for inputs, target in og_dataloader:
            data = pd.DataFrame(target, columns=self.experiment_x.dataloader.data_with_sensitive.columns)
            if args.attr_to_gen.lower() is not 'none':
                for col in data.columns:
                    if args.attr_to_gen.lower() in col.lower():
                        data.drop([col], axis=1, inplace=True)
            target = torch.tensor(data.values.astype(np.float32))
            loss_dim_batch = san_loss_fn(inputs.float(), target.float())
            loss_dim = loss_dim_batch.mean(dim=0)
            self.sanitized_loss = loss_dim
            break
        
        # here we compare the above with L1 from data generated by reconstructor
        #   contained in self.lowest_loss_per_dim
        is_better = []
        how_much = []
        better_count = 0
        for i, loss in enumerate(self.sanitized_loss):
            how_much.append((loss - self.lowest_loss_per_dim[i]))
            if self.lowest_loss_per_dim[i] < loss:
                is_better.append(True)
                better_count = better_count+1
            else:
                is_better.append(False)
                
        if args.input_dataset == 'gansan':
            is_better_df = pd.DataFrame([self.sanitized_loss, self.lowest_loss_per_dim, is_better,how_much], columns=self.experiment_x.data_pp.encoded_features_order)
        else: 
            is_better_df = pd.DataFrame([self.sanitized_loss, self.lowest_loss_per_dim, is_better,how_much], columns=self.experiment_x.data_pp.cols_order)

        # Save model meta data in txt file
        with open(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}.txt", 'w+') as f:
            f.write(f"Epochs: {self.num_epochs} \n \n")
            f.write(f"Lowest lost Generated:\n {self.lowest_loss_per_dim} \n \n")
            f.write(f"Loss from sanitized data: \n{self.sanitized_loss.numpy().tolist()} \n")
            f.write(f"\nTotal of {better_count} / {len(is_better_df.columns)} are now closer to original data (L1 distance).")
            f.write(f"\n \n Learning Rate: {self.learning_rate} \n")
            f.write(f"Number Epochs: {self.num_epochs} \n")
            f.write(f"weight decay: {self.wd}\n")
            f.write(f"Training Loss: {str(self.train_loss)}\n")
            f.write(f"Test Loss: {str(self.test_loss_fn)} \n")
            f.write(f"self.self.optimizer: {str(self.optimizer)}\n")
            f.write(f"Model Architecture: {self.model}\n")
            f.write(f"Training completed in: {(end-start)/60:.2f} minutes\n")
            f.write(f"Train loss values: {str(self.ave_train_loss)} \n")
            f.write(f"Test loss values: {str(self.test_accuracy)}\n")

        os.mkdir(path_base)
        is_better_df.to_csv(path_base+f"reconstruction_appraisal.csv", index=False)

    def post_training_metrics(self):
        '''
            This will calculate (1) diversity within generated dataset, 
                and the (2) damage the generated dataset has
                    Those will be compared to both the original and sanitized        

            ISSUE -> Sanitized data does not have sensitive attribute, hence fewer dimensions    
        '''
        # Need to created a Encoder object with original data just in order to have matching columns when calculating damage and Diversity
        print(f"Starting calculation Three-way of Diversity, Damage and graphs: {self.dim_red}")
        start = time.time()


        # TODO loop through those 3 paragraphs
        # Sanitized data == model input data
        test_san = self.experiment_x.dataloader.df_data.values[self.experiment_x.dataloader.train_size:,:] 
        headers = self.experiment_x.dataloader.headers_wo_sensitive
        san_data = pd.DataFrame(test_san, columns=headers)
        san_data.to_csv(model_saved+'junk_test_san.csv', index=False)
        san_data_rough = pd.read_csv(model_saved+'junk_test_san.csv')
        self.san_encoder = d.Encoder(san_data_rough)
        pd_san_data = self.san_encoder.df

        # Original data == Model's target data
        test_og = self.experiment_x.dataloader.df_label.values[self.experiment_x.dataloader.train_size:,:]
        headers = self.experiment_x.data_pp.encoded_features_order
        og_data = pd.DataFrame(test_og, columns=headers)
        og_data.to_csv(model_saved+'junk_test_og.csv',  index=False)
        og_data_rough = pd.read_csv(model_saved+'junk_test_og.csv')
        self.og_encoder = d.Encoder(og_data_rough)
        self.og_encoder.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
        pd_og_data = self.og_encoder.df

        #Generated Data
        test_gen = self.best_generated_data
        test_gen.to_csv(model_saved+'junk_test_gen.csv', index=False)
        gen_data_rough = pd.read_csv(model_saved+'junk_test_gen.csv')
        self.gen_encoder = d.Encoder(gen_data_rough)
        self.gen_encoder.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
        pd_gen_data = self.gen_encoder.df

        test_data_dict = {}
        test_data_dict['orig_san'] = [pd_og_data, pd_san_data]
        test_data_dict['san_gen'] = [ pd_san_data,pd_gen_data]
        test_data_dict['gen_orig'] = [pd_gen_data,pd_og_data] 
        
        # Calculate diversity among three dataset
        # Diversity is line by line, a
        path_to_eval = f"{path_to_exp}model_metrics/"
        os.mkdir(path_to_eval)

        diversities = []
        saver_dict = {}
        # for key in test_data_dict:
        #     # Save all damage and diversity results
        #     saver_dict[key] = r.DiversityDamageResults(resultDir=path_to_eval, result_file=f"cat_damage_{key}.csv",  num_damage=f"numerical_damage_{key}.csv", overwrite=True)

        #     div = at.Diversity()
        #     diversity = div(test_data_dict[key][0], f"original_{key.split('_')[0]}")
        #     diversity.update(div(test_data_dict[key][1], f"transformed_{key.split('_')[1]}"))
        #     diversities.append(diversity)
        # DECODE DATA
        # Sanitized data == model input data
        # self.san_encoder.inverse_transform()
        # pd_san_data = self.san_encoder.df
        # # Original data == Model's target data
        # self.og_encoder.inverse_transform()
        # pd_og_data = self.og_encoder.df
        # # Generated data
        # self.gen_encoder.inverse_transform()
        # pd_gen_data = self.gen_encoder.df
        # pd_gen_data.to_csv(f"{model_saved}lowest-loss-generated-data_ep{self.lowest_loss_ep}.csv",  index=False)

        #TODO Figure out how to update a dict
        # test_data_dict['orig_san'] = [pd_og_data,  pd_san_data]
        # test_data_dict['san_gen'] = [ pd_san_data, pd_gen_data]
        # test_data_dict['gen_orig'] = [pd_gen_data, pd_og_data]
        
        # dam_dict = {}
        # i = 0

        # Calculation of Damage; Damage is for each feature, compared across two datasets
        # for key in test_data_dict:
        #     dam = at.Damage()

        #     d_cat, d_num = dam(original=test_data_dict[key][0], transformed=test_data_dict[key][1])
        #     dam_dict[key] = [d_cat, d_num]

        #     saver_dict[key].add_results(diversity=diversities[i], damage_categorical=d_cat, damage_numerical=d_num, epoch=self.num_epochs, alpha_=args.alpha)
        #     i = i +1
            
        #     # Calculate dimention reduction is required
        #     if (self.dim_red).lower() != 'none':
        #         # each color is each label as specific
        #         dr = at.DimensionalityReduction()
        #         dr.clusters_original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]},labels=test_data_dict[key][0]['sex'], dimRedFn=self.dim_red, savefig=path_to_eval+f"label_{key}_{self.dim_red}.png")

        #         # Each color is a dataset
        #         dr.original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]}, dimRedFn=self.dim_red, savefig=path_to_eval+f"dataset_{key}_{self.dim_red}.png")

        # os.remove(model_saved+'junk_test_og.csv')
        # os.remove(model_saved+'junk_test_san.csv')
        # os.remove(model_saved+'junk_test_gen.csv')
        end = time.time()
        

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
        plt.savefig(path_to_exp+f"{args.exp_name}-{a}_train-loss.png")
    
    def pandas_describe(self):
        '''
            Runs and save pandas.describe function on all three dataset:
            Orginal, Sanitized and Generated, then saves under /base_metrics
        '''
        a = self.og_encoder.df.describe()
        b = self.san_encoder.df.describe()
        c = self.gen_encoder.df.describe()
        
        a.to_csv(path_base+"original.csv")
        b.to_csv(path_base+"sanitized.csv")
        c.to_csv(path_base+"generated.csv")

def main():

    # Pre Process Data
    experiment = PreProcessing()

    # Create training instance
    training_instance = Training(experiment)

    # Train the AE
    training_instance.train_model()

    # Run Damage, Diversity and dimension reduction graphs Calculation
    training_instance.post_training_metrics()

    # Generate test and train loss graphs (L1)
    training_instance.gen_loss_graphs()
    training_instance.pandas_describe()

    print(f"\n \n Experiment can be found under {path_to_exp} \n \n ")

# Setup folders and global variables
parser = argparse.ArgumentParser()
args, path_to_exp, model_saved = utils.parse_arguments(parser)
path_base = path_to_exp+'distance_comparison/'

# Launch
main()
