from typing import List
import pandas as pd
import os


def dummy_encode(df: pd.core.frame.DataFrame, cat_cols : List[str] )-> pd.core.frame.DataFrame:
    '''
        Will find columns from second parameter that corresponds to 
        column names in pandas and use dummy encoding scheme from
        pandas to encode, then return the new dataframe

        :param df: Dataframe to work with
        :param cat_cols: list of columns that needs to be changed 

        :return same dataframe, encoded if required

    '''
    num_rows_original = df.shape[0]
    if cat_cols == 0:
        return df
    else:
        new_df = pd.get_dummies(df, columns=cat_cols)
        num_rows_new = new_df.shape[0]
        if num_rows_original != num_rows_new:
            print(f"\n There might to have been data loss during encoding. \n Row counts do not match \n")
            
        return new_df

def find_cat(df: pd.core.frame.DataFrame) -> str:
    '''
        Iterates through columns to find ones that contains strings
         ** Only looks at first value, hence assumes all column is 
                 of same type..

        df: pandas dataframe to search in
    '''
    cat_cols = []
    for col_name, col_data in df.iteritems():
        if type(col_data.values[0]) == str:
            cat_cols.append(col_name)

    return cat_cols

def adjust_enc_errors(df: pd.core.frame.DataFrame, df2: pd.core.frame.DataFrame) -> (pd.core.frame.DataFrame, pd.core.frame.DataFrame):
        '''
            When encoding with get_dummies, data and labels might return different values
            since the some domain in each column might not be exactly the same
            Solution is to add the missing values in the each pandas
        '''
        missing_cols = set( df.columns ) - set(df2.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            df2[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        labels_df = df2[df.columns]

#         # then the other way around
#         missing_cols = set( df2.columns ) - set( df.columns )
#         # Add a missing column in test set with default value equal to 0
#         for c in missing_cols:
#             df[c] = 0
#         # Ensure the order of column in the test set is in the same order than in train set
#         data_df = df[df2.columns]

        return df, labels_df

def check_dir_path(path_to_check: str) -> str:
    '''
        Checks if provided path is currently a at current level directory.
        If it is, it appends a number to the end and checks again 
        until no directory with such name exists

        path_to_check: str The path to location to check

        return: str New path with which os.mkdir can be called
    '''
    if os.path.isdir(path_to_check):
        print("Experiment with name: \'{}\' already exists. Appending int to folder name \n \n ".format(path_to_check))
        if os.path.isdir(path_to_check):
            expand = 1
            while True:
                expand += 1
                new_path = path_to_check[:-1] + '_' + str(expand) + '/'
                if os.path.isdir(new_path):
                    continue
                else:
                    break
    return new_path