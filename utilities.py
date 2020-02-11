from typing import List
import pandas as pd


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
