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
    parser.add_argument('-a','--alpha', type=float, default=0.2, help='Value of alpha used when  sanitizing the dataset we use as input.', required=False, choices=[0.2,0.8,0.9875])
    parser.add_argument('-gen','--attr_to_gen', type=str, default='none', help='Attribute that we want to remove from input if present, and have model infer; out_dim=in_dim+2', required=False)
    parser.add_argument('-in','--input_dataset', type=str, default='gansan', help='Dataset to use as input. Currently support `gansan` and `disp_impact`', required=False, choices=['gansan', 'disp_impact'])
    parser.add_argument('-ns','--no_sex', type=str, default=True, help='Whether `sex` column is taken into account in training the reconstructor. Should be no', required=False)
    parser.add_argument('-kf','--kfold', type=str, default='false', help='Whether classifiers used after training to predict sensitive should use kfold')

    # parser.add_argument('--discriminator_size', type=tuple, default=(256, 128, 1), help='The dimension size of the discriminator. (default value: (256, 128, 1))')
    args = parser.parse_args()
    if_alpha = f"_{args.alpha}a" if args.input_dataset == 'gansan' else ""
    exp_name = f"{args.input_dataset}{if_alpha}_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_" +args.exp_name
    path_to_exp = check_dir_path(f'./experiments/{exp_name}/')
    os.mkdir(path_to_exp)
    model_saved = path_to_exp+'models_data/'
    os.mkdir(model_saved)

    return args, path_to_exp, model_saved


    # def post_training_metrics(self):
    #     '''
    #         This will calculate (1) diversity within generated dataset, 
    #             and the (2) damage the generated dataset has
    #                 Those will be compared to both the original and sanitized        

    #         ISSUE -> Sanitized data does not have sensitive attribute, hence fewer dimensions    
    #     '''
    #     # Need to created a Encoder object with original data just in order to have matching columns when calculating damage and Diversity
    #     # print(f"Starting calculation Three-way of Diversity, Damage and graphs: {self.dim_red}")
    #     # start = time.time()


        # # TODO loop through those 3 paragraphs
        # # Sanitized data == model input data
        # test_san = self.experiment_x.dataloader.df_data.values[self.experiment_x.dataloader.train_size:,:] 
        # headers = self.experiment_x.dataloader.headers_wo_sensitive
        # san_data = pd.DataFrame(test_san, columns=headers)
        # san_data.to_csv(model_saved+'junk_test_san.csv', index=False)
        # san_data_rough = pd.read_csv(model_saved+'junk_test_san.csv')
        # self.san_encoder = d.Encoder(san_data_rough)
        # pd_san_data = self.san_encoder.df

        # # Original data == Model's target data
        # test_og = self.experiment_x.dataloader.df_label.values[self.experiment_x.dataloader.train_size:,:]
        # headers = self.experiment_x.dataloader.data_with_sensitive.columns
        # og_data = pd.DataFrame(test_og, columns=headers)
        # og_data.to_csv(model_saved+'junk_test_og.csv',  index=False)
        # og_data_rough = pd.read_csv(model_saved+'junk_test_og.csv')
        # self.og_encoder = d.Encoder(og_data_rough)
        # self.og_encoder.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
        # pd_og_data = self.og_encoder.df
        
        # #Generated Data
        # test_gen = self.best_generated_data
        # headers = self.experiment_x.dataloader.data_with_sensitive.columns
        # test_gen = pd.DataFrame(test_gen, columns=headers)
        # test_gen.to_csv(model_saved+'junk_test_gen.csv', index=False)
        # gen_data_rough = pd.read_csv(model_saved+'junk_test_gen.csv')
        # self.gen_encoder = d.Encoder(gen_data_rough)
        # self.gen_encoder.load_parameters(path_to_exp, prmFile=f"{args.input_dataset}_parameters_data.prm")
        # pd_gen_data = self.gen_encoder.df

        # test_data_dict = {}
        # test_data_dict['orig_san'] = [pd_og_data, pd_san_data]
        # test_data_dict['san_gen'] = [ pd_san_data,pd_gen_data]
        # test_data_dict['gen_orig'] = [pd_gen_data,pd_og_data] 
        
        # Calculate diversity among three dataset
        # Diversity is line by line, a
        # path_to_eval = f"{path_to_exp}model_metrics/"
        # os.mkdir(path_to_eval)

        # diversities = []
        # saver_dict = {}
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
        # end = time.time()