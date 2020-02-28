import enum
import copy
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler
import pickle



def rm_qmark(df_data: pd.DataFrame, rm_row=True) -> pd.DataFrame:
    '''
        removes question mark string value from dataframe and either delete the row or replaces value with None type values.
        It first moves it in numpy array for faster computation.
        Especially usefull for Adult Dataset

        :PARAMS
        self explanatory
        rm_rows: will remove whole row if True. Else will change value for None
    '''
    a = df_data.to_numpy()
    headers = df_data.columns
    for row_index, row in enumerate(a):
        for col_index, data in enumerate(row):
            if data =='?':
                if rm_row==True:
                    np.delete(a,[row_index],axis=0)
                else:
                    a[row_index,col_index]= None 
                
    newdf = pd.DataFrame(a, columns=headers)
    return newdf

class Encoder:
    """
    Class to handle all the preprocessing steps before passing the dataset to pytorch
    """

    def __init__(self, csv, numeric_as_categoric_max_thr=5, scale=0, prep_excluded=None, prep_included=None):
        """

        :param csv: the path to the dataset
        :param numeric_as_categoric_max_thr: Maxiimum number of values for a numeric column to be considered as
        categoric
        :param scale: the lower bound of the feature scaling. Either 0 or -1
        :param prep_excluded: Columns to exclude from the preprocessing
        :param prep_included: Columns to include from the preprocessing. This overwrite the prep_excluded
        """
        if isinstance(csv, str):
            self.df = pd.read_csv(csv)
        elif isinstance(csv, pd.DataFrame):
            self.df = csv.copy(True)
        else:
            raise ValueError("Invalid input type: {}".format(csv))

        self.cols_order = self.df.columns
        self.prep_excluded = prep_excluded
        self.prep_included = prep_included
        if prep_included is not None:
            self.prep_excluded = list(set(self.cols_order) - set(prep_included))
        self.encoded_features_order = None
        self.categorical_was_numeric = {}
        self.num_as_cat = numeric_as_categoric_max_thr
        self.scale = scale
        self.scaler = MinMaxScaler(feature_range=(scale, 1))
        self.cat_clm = None
        self.num_clm = None
        self.int_clm = None

    def save_parameters(self, prmPath, prmFile="parameters.prm"):
        """
        Dump the parameters on the disc
        :param prmPath: location where to save params
        """
        o = {
            "cols_order": self.cols_order,
            "prep_excluded": self.prep_excluded,
            "prep_included": self.prep_included,
            "encoded_features_order": self.encoded_features_order,
            "categorical_was_numeric": self.categorical_was_numeric,
            "num_as_cat": self.num_as_cat,
            "scale": self.scale,
            "cat_clm": self.cat_clm,
            "num_clm": self.num_clm,
            "int_clm": self.int_clm,
            "scaler": self.scaler,
        }
        pickle.dump(o, open("{}/{}".format(prmPath, prmFile), "wb"))

    def load_parameters(self, prmPath, prmFile="parameters.prm"):
        o = pickle.load(open("{}/{}".format(prmPath, prmFile), "rb"))
        self.cols_order = o["cols_order"]
        self.prep_excluded = o["prep_excluded"]
        self.prep_included = o["prep_included"]
        self.encoded_features_order = o["encoded_features_order"]
        self.categorical_was_numeric = o["categorical_was_numeric"]
        self.num_as_cat = o["num_as_cat"]
        self.scale = o["scale"]
        self.cat_clm = o["cat_clm"]
        self.num_clm = o["num_clm"]
        self.int_clm = o["int_clm"]
        self.scaler = o["scaler"]

    def __filter_features__(self, columns, prefix_sep="="):
        """
        Overriding the filter function of pandas, since we might have one-hot encoded columns.
        :param df: dataframe from which to remove columns
        :param columns: the columns to remove
        :param prefix_sep: the prefix used to separate attributes and their respective values when columns are encoded.
        :return: the filtered columns
        """
        cls = []
        if isinstance(columns, str):
            columns = [columns]
        for c in columns:
            # Handling encoded columns. Encoded attribute name.
            cls.extend(self.df.filter(regex="^{}{}".format(c, prefix_sep)).columns.tolist())
            # Handling other columns while avoiding to mistook some. Attribute name only.
            cls.extend(self.df.filter(regex="^{}$".format(c, )).columns.tolist())
        return cls

    def __find_categorical__(self):
        """
        List int and float columns
        :return: The list of num column
        """
        cat_clm = []
        num_clm = []
        for c in self.df.select_dtypes(include=["int", 'float', 'double']).columns:
            if self.df[c].value_counts().shape[0] <= self.num_as_cat:
                self.categorical_was_numeric.update({c: self.df[c].dtype})
                cat_clm.append(c)
            else:
                num_clm.append(c)
        cat_clm.extend(self.df.select_dtypes(exclude=["int", 'float', 'double']).columns.tolist())
        self.cat_clm = cat_clm
        self.num_clm = num_clm
        self.int_clm = self.df.select_dtypes(include=["int"]).columns.tolist()

    def __from_dummies__(self, prefix_sep='=', **kwargs):
        """
            Convert encoded columns into original ones
        """
        if 'ext_data' in kwargs:
            data = kwargs['ext_data']
        else:
            data = self.df
        data = rm_qmark(data)
        categories = self.cat_clm
        cat_was_num = self.categorical_was_numeric
        out = data.copy()
        for l in categories:
            cols = data.filter(regex="^{}{}".format(l, prefix_sep), axis=1).columns
            labs = [cols[i].split(prefix_sep)[-1] for i in range(cols.shape[0])]
            out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
            if l in cat_was_num.keys():
                out[l] = out[l].astype(cat_was_num[l])
        if 'ext_data' in kwargs:
            return out
        else:
            self.df = out

    def __squash_in_range__(self):
        """
        Squash df values between min and max from scaler
        """
        for c in self.num_clm:
            i = self.df.columns.get_loc(c)
            self.df.loc[self.df[c] > self.scaler.data_max_[i], c] = self.scaler.data_max_[i]
            self.df.loc[self.df[c] < self.scaler.data_min_[i], c] = self.scaler.data_min_[i]

    def set_encoded_features_order(self, features_order=None, from_df=False):
        if from_df:
            self.encoded_features_order = self.df.columns.tolist()
        else:
            self.encoded_features_order = features_order

    def set_features_params(self, features_order, categorical_was_numeric, num_clm, cat_clm):
        """
        Set the original feature ordering, and the original type of numeric columns that have been considered
        as categorical because of they had a limited number of values
        """
        self.encoded_features_order = features_order
        self.categorical_was_numeric = categorical_was_numeric
        self.num_clm = num_clm
        self.cat_clm = cat_clm

    def __encoded_features_format__(self):
        """
        Add missing columns to have the same dataset structure
        scale should be the lower scale as given i
        """
        prep_excluded = self.prep_excluded if self.prep_excluded is not None else []
        if self.encoded_features_order is not None:
            for c in self.encoded_features_order:
                if c not in self.df.columns and c not in prep_excluded:
                    self.df[c] = self.scale

            self.df = self.df[self.encoded_features_order]

    def __round_integers__(self):
        """
        Round the columns that where of type integer to begin with
        """
        for c in self.int_clm:
            self.df[c] = self.df[c].round().astype("int")

    def fit_transform(self, prefix_sep='='):
        """
        Apply all transformation. Some attribute will be set. If you do not want to change values of those attributes,
        call transform
        """
        excluded = pd.DataFrame()
        # if self.prep_included is not None and len(self.prep_included) > 0:
        #     excluded = self.df[self.prep_included] # Included
        #     self.df = self.df.drop(self.prep_included, axis=1) # Excluded
        #     excluded, self.df = self.df, excluded # Inverting both
        # else:
        if self.prep_excluded is not None:
            excluded = self.df[self.prep_excluded]
            self.df.drop(self.prep_excluded, axis=1, inplace=True)
        self.__find_categorical__()
        self.df = pd.get_dummies(self.df, columns=self.cat_clm, prefix_sep=prefix_sep)

        # Scale the data, then add missing columns and set a fixed order, then add the excluded columns. Excluded is at
        # the end of the processing as we are not supposed to touch them.
        # columns
        # Extract columns order as they are encoded
        self.set_encoded_features_order(from_df=True)
        # Scaler contains all columns except the excluded ones
        # self.df.iloc[:, :] = self.scaler.fit_transform(self.df.values)
        self.df = pd.DataFrame(self.scaler.fit_transform(self.df.values), columns=self.df.columns)
        self.df = pd.concat([self.df, excluded], axis=1)

        # Complete missing columns
        # self.__features_formatting__()
        # Scaler contains all columns
        # self.df.iloc[:, :] = self.scaler.fit_transform(self.df.values)
        # self.df = pd.DataFrame(self.scaler.fit_transform(self.df.values), columns=self.df.columns)
        # self.df = pd.concat([self.df, excluded], axis=1)

    def transform(self, prefix_sep='='):
        """
        Transform dataset without setting values.
        """
        excluded = pd.DataFrame()
        if self.prep_excluded is not None:
            excluded = self.df[self.prep_excluded]
            self.df.drop(self.prep_excluded, axis=1, inplace=True)
        self.df = pd.get_dummies(self.df, columns=self.cat_clm, prefix_sep=prefix_sep)
        # Complete missing columns, and give a standard column order.
        self.__encoded_features_format__()
        self.df = pd.DataFrame(self.scaler.transform(self.df.values), columns=self.df.columns)
        self.df = pd.concat([self.df, excluded], axis=1)

    def inverse_transform(self):
        """
            Recover the original data
        """
        excluded = pd.DataFrame()
        # if self.prep_included is not None and len(self.prep_included) > 0:
        #     excluded = self.df[self.prep_included] # Included
        #     self.df = self.df.drop(self.prep_included, axis=1) # Excluded
        #     excluded, self.df = self.df, excluded # Inverting both
        # else:
        if self.prep_excluded is not None:
            excluded = self.df[self.prep_excluded]
            self.df.drop(self.prep_excluded, axis=1, inplace=True)
        self.df = pd.DataFrame(self.scaler.inverse_transform(self.df.values), columns=self.df.columns)
        self.__from_dummies__()
        # Scaler contains all columns
        self.__squash_in_range__()
        self.__round_integers__()
        self.df = pd.concat([self.df, excluded], axis=1)[self.cols_order]

    def copy(self, deepcopy=False):
        if deepcopy:
            return copy.deepcopy(self)
        return copy.copy(self)


#### No need for alternating class samples. Just make a good stratification when splitting sets, and
#### Use stratify in torch dataloader. Always read all function params. You might have some surprises if not.
class ProportionsMatching:
    """ Class to match proportions of dataset, alternate values such that each group of values match a the dataset
     proportions """

    def __init__(self, df, features_name, features_positive_values):
        """
        :param df: Dataframe on which we want to match proportions.
        :param features_name: List of features, note that the last one will be with the most accurately set
        :param features_positive_values: list of positive values for each feature name.
        """
        self.df = df
        self.features_name = features_name
        self.features_positive_values = features_positive_values

    def __call__(self, *args, **kwargs):
        for fn, fpv in zip(self.features_name, self.features_positive_values):
            self.__make__(fn, fpv)
        return self.df

    def __make__(self, feature_name, feature_value):
        """
        Do the procedure.
        :param feature_name: Name of the feature to consider
        :param feature_value: positive value of the attribute name.
        """
        mask = data[feature_name] == feature_value
