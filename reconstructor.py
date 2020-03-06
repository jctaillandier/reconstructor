# Personal Imports
from Modules import analysisTools as at
from Modules import utilities as utils
from Modules import datasets as d
from Modules import results as r

# from Stats import Plots as Pl
import csv, sys, yaml, math, json, tqdm, time, torch, random, os.path, warnings, argparse, importlib, torchvision 
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

a = 'adult'
# if not sys.argv:
#     raise EnvironmentError("Not enough parameters added to command: Need (1) paramsfile.yaml and (2) experiment_name")
# args = sys.argv[1:]

# if args[0][-5:] != '.yaml':
#     raise ValueError("first argument needs to be the parameters file in yaml format")
# params_file = args[0]

# if args[1]:
#     exp_name = args[1]
# else:
#     print("No name given to experiment.")
#     exp_name = "no-name"

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
        self.batchSize = args.batch_size       
        test_batch_size = args.test_batch_size       
        percent_train_set = args.percent_train_set
        self.col_to_rm = self.params['model_params']['col_to_del']['value']

        df2 = pd.read_csv(import_path)
        self.df_labels = pd.read_csv(label_path)
        df2.replace('?', None, inplace=True)
        self.df_labels.replace('?', None, inplace=True)

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

        self.dataloader = My_dataLoader(self.batchSize, f"{import_path[:-4]}_NoCat.csv", f"{label_path[:-4]}_NoCat.csv", self.n_test, test_batch_size=test_batch_size, col_to_rm=self.col_to_rm)

        self.data_dataframe = self.data_pp.df
        self.labels_dataframe = self.labels_pp.df

class My_dataLoader:
    def __init__(self, batch_size : int, data_path :str, label_path:str, n_train :int, test_batch_size:int=128, col_to_rm:str='None'):
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

        # if col_to_rm is not 'None':
        #     del df_data[str(col_to_rm)]
        
        self.a = df_data.values
        self.b = df_label.values
        self.trdata = torch.tensor(self.a[:self.train_size,:]) # where data is 2d [D_train_size x features]
        self.trlabels = torch.tensor(self.b[:self.train_size,:]) # also has too be 2d
        self.tedata = torch.tensor(self.a[self.train_size:,:]) # where data is 2d [D_train_size x features]
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
        # self.decoder = nn.Sequential(
        #     # batchnorm,
        #     nn.Linear(in_dim, in_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_dim, in_dim),
        #     nn.LeakyReLU(), 
        #     nn.Linear(in_dim, out_dim)
        # )

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
    
    test_loss = 0
    test_size = 0
    batch_ave = 0
    with torch.no_grad():
        for inputs, target in experiment.dataloader.test_loader:
            
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs.float())
            np_output = output.cpu().numpy()
            headers = experiment.data_pp.encoded_features_order
            gen_data = pd.DataFrame(np_output, columns=headers)

            test_size = len(inputs.float())
            test_loss += test_loss_fn(output.float(), target.float()).item() 
    
    return test_loss, gen_data 

class Training:
    def __init__(self, experiment_x: PreProcessing, model_type:str='autoencoder'):
        '''
            Training class takes in the Preprocessing object containing all data and creates an instance of training that is loading all hyper parameters from the specified file.
            It will allow the training and testing loops, as well as the generation of all relevant metrics for comparison and analysis post-training

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
        self.best_generated_data = []
        self.lowest_loss_ep = -1

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch} of {self.num_epochs} running...")

            # Iterate on train set with SGD (adam)
            batch_ave_tr_loss = train(self.model,self.experiment_x.dataloader.train_loader, self.optimizer, self.train_loss)
            self.ave_train_loss.append(batch_ave_tr_loss.cpu().numpy().item())

            # Check test set metrics (+ generate data if last epoch )
            loss, model_gen_data = test(self.model, self.experiment_x, self.test_loss_fn)

            if loss < lowest_test_loss:
                if os.path.isfile(model_saved+f"lowest-test-loss-model_ep{self.lowest_loss_ep}.pth"):
                    os.remove(model_saved+f"lowest-test-loss-model_ep{self.lowest_loss_ep}.pth")
                lowest_test_loss = loss
                fm = open(model_saved+f"lowest-test-loss-model_ep{epoch}.pth", "wb")
                torch.save(self.model.state_dict(), fm)
                fm.close()
                self.lowest_loss_ep = epoch
                # Then we want this to be used as generated data
                self.best_generated_data = model_gen_data

            print(f"Epoch {epoch} complete. Test Loss: {loss:.6f} \n")      
            self.test_accuracy.append(loss)#/len(self.experiment_x.dataloader.test_loader.dataset))

        fm = open(model_saved+f"final-model_{self.num_epochs}ep.pth", "wb")
        torch.save(self.model.state_dict(), fm)

        end = time.time()
        print(f"Training on {self.num_epochs} epochs completed in {(end-start)/60} minutes.\n")

        # Save model meta data in txt file
        with open(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}.txt", 'w+') as f:
            f.write(f"Epochs: {self.num_epochs} \n")
            f.write(f"Learning Rate: {self.learning_rate} \n")
            f.write(f"Number Epochs: {self.num_epochs} \n")
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
        print(f"Starting calculation Three-way of Diversity, Damage and visual: {self.dim_red}")
        start = time.time()

        
        # TODO loop through those 3 paragraphs
        # Sanitized data == model input data
        test_san = self.experiment_x.dataloader.a[self.experiment_x.dataloader.train_size:,:] 
        headers = self.experiment_x.data_pp.encoded_features_order
        san_data = pd.DataFrame(test_san, columns=headers)
        san_data.to_csv(model_saved+'junk_test_san.csv', index=False)
        san_data_rough = pd.read_csv(model_saved+'junk_test_san.csv')
        self.san_encoder = d.Encoder(san_data_rough)
        self.san_encoder.load_parameters(path_to_exp, prmFile="parameters_data.prm")
        pd_san_data = self.san_encoder.df

        # Original data == Model's target data
        test_og = self.experiment_x.dataloader.b[self.experiment_x.dataloader.train_size:,:]
        headers = self.experiment_x.data_pp.encoded_features_order
        og_data = pd.DataFrame(test_og, columns=headers)
        og_data.to_csv(model_saved+'junk_test_og.csv',  index=False)
        og_data_rough = pd.read_csv(model_saved+'junk_test_og.csv')
        self.og_encoder = d.Encoder(og_data_rough)
        self.og_encoder.load_parameters(path_to_exp, prmFile="parameters_data.prm")
        pd_og_data = self.og_encoder.df

        #Generated Data
        test_gen = self.best_generated_data
        test_gen.to_csv(model_saved+'junk_test_gen.csv', index=False)
        gen_data_rough = pd.read_csv(model_saved+'junk_test_gen.csv')
        self.gen_encoder = d.Encoder(gen_data_rough)
        self.gen_encoder.load_parameters(path_to_exp, prmFile="parameters_data.prm")
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
        for key in test_data_dict:
            # Save all damage and diversity results
            saver_dict[key] = r.DiversityDamageResults(resultDir=path_to_eval, result_file=f"cat_damage_{key}.csv",  num_damage=f"numerical_damage_{key}.csv", overwrite=True)

            div = at.Diversity()
            diversity = div(test_data_dict[key][0], f"original_{key.split('_')[0]}")
            diversity.update(div(test_data_dict[key][1], f"transformed_{key.split('_')[1]}"))
            diversities.append(diversity)

        # DECODE DATA
        # Sanitized data == model input data
        self.san_encoder.inverse_transform()
        pd_san_data = self.san_encoder.df
        # Original data == Model's target data
        self.og_encoder.inverse_transform()
        pd_og_data = self.og_encoder.df
        # Generated data
        self.gen_encoder.inverse_transform()
        pd_gen_data = self.gen_encoder.df
        pd_gen_data.to_csv(f"{model_saved}lowest-loss-generated-data_ep{self.lowest_loss_ep}.csv",  index=False)

        #TODO Figure out how to update a dict
        test_data_dict['orig_san'] = [pd_og_data,  pd_san_data]
        test_data_dict['san_gen'] = [ pd_san_data, pd_gen_data]
        test_data_dict['gen_orig'] = [pd_gen_data, pd_og_data]
        
        dam_dict = {}
        i = 0

        # Calculation of Damage; Damage is for each feature, compared across two datasets
        for key in test_data_dict:
            dam = at.Damage()

            d_cat, d_num = dam(original=test_data_dict[key][0], transformed=test_data_dict[key][1])
            dam_dict[key] = [d_cat, d_num]

            saver_dict[key].add_results(diversity=diversities[i], damage_categorical=d_cat, damage_numerical=d_num, epoch=self.num_epochs, alpha_=args.alpha)
            i = i +1
            
            # Calculate dimention reduction is required
            if (self.dim_red).lower() != 'none':
                # each color is each label as specific
                dr = at.DimensionalityReduction()
                dr.clusters_original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]},labels=test_data_dict[key][0]['sex'], dimRedFn=self.dim_red, savefig=path_to_eval+f"label_{key}_{self.dim_red}.png")

                # Each color is a dataset
                dr.original_vs_transformed_plots({key.split('_')[0]: test_data_dict[key][0], key.split('_')[1]: test_data_dict[key][1]}, dimRedFn=self.dim_red, savefig=path_to_eval+f"dataset_{key}_{self.dim_red}.png")

        os.remove(model_saved+'junk_test_og.csv')
        os.remove(model_saved+'junk_test_san.csv')
        os.remove(model_saved+'junk_test_gen.csv')
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
        plt.savefig(path_to_exp+f"{str.replace(time.ctime(), ' ', '_')}-{a}_train-loss.png")
    
    def my_metrics(self):
        # Std-dev and mean for each encoded columns
        self.og_encoder.fit_transform()
        self.san_encoder.fit_transform()
        self.gen_encoder.fit_transform()
        headers = self.san_encoder.df.columns

        stats = {}
        stats['og_mean'] = [self.og_encoder.df.mean(axis=0), self.og_encoder.df.std()]
        stats['san_mean'] = [self.san_encoder.df.mean(axis=0),self.san_encoder.df.std()]
        stats['gen_mean'] = [self.gen_encoder.df.mean(axis=0),self.gen_encoder.df.std()]
        path_base = path_to_exp+'base_metrics/'
        os.mkdir(path_base)

        mean = pd.concat([stats['og_mean'][0],stats['san_mean'][0],stats['gen_mean'][0]])
        std_dev = pd.concat([stats['og_mean'][1],stats['san_mean'][1],stats['gen_mean'][1]])
        # mean.insert(loc=0,column='Dataset', value=['Title','Original Data', 'Sanitized Data', 'Generated Data'])
        # std_dev.insert(loc=0,column='Dataset', value=['Title','Original Data', 'Sanitized Data', 'Generated Data'])
        
        mean.to_csv(path_base+'mean.csv',index=False)
        std_dev.to_csv(path_base+'std_dev.csv',index=False)

if __name__ == '__main__':

    # Setup folders and global variables
    parser = argparse.ArgumentParser()
    args, path_to_exp, model_saved = utils.parse_arguments(parser)
    params_file = args.params

    # Pre Process Data
    experiment = PreProcessing(args.params)

    # Create training instance
    training_instance = Training(experiment)

    # Train the AE
    training_instance.train_model()

    # Run Metrics Calculation
    training_instance.post_training_metrics()

    # Generate test and train loss graphs (L1)
    training_instance.gen_loss_graphs()
    # training_instance.my_metrics()

    print(f"\n \n Experiment can be found under {path_to_exp} \n \n ")

