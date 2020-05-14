import numpy as np
from Modules import datasets as d
from Modules import Visualisation
from scipy.spatial.distance import cdist
import pandas as pd


class Diversity: 
    """
         Compute the diversity between in the given dataset. We assume the dataset to be encoded. 
         Compares on a row by row basis
    """

    def __init__(self, diversity_fn=None):
        """ Initialization
        :param diversity_fn: function to compute diversity with. If None, will use the default function """
        self.diversity_fn = diversity_fn if diversity_fn is not None else self.default_diversity

    def default_diversity(self, data: pd.core.frame.DataFrame, suffix=None):
        """ 
            default diversity function. Data is
            """
        if isinstance(data, d.Encoder):
            data = data.df
        try:
            distance = cdist(data, data, "euclidean").sum(1)
            distance /= np.sqrt(data.shape[1]) * (distance.shape[0] - 1)
        except MemoryError:
            distance = []
            for i in range(data.shape[0]):
                distance.append(np.sqrt(((data.iloc[i] - data) ** 2).sum(1)).sum())
            distance = np.array(distance) / ((data.shape[0] - 1) * np.sqrt(data.shape[1]))

        distance = distance.sum() / distance.shape[0]
        key = "diversity"
        if suffix is not None:
            key += "_{}".format(suffix)
        return {key: [distance]}

    def __call__(self, *args, **kwargs):
        return self.diversity_fn(*args, **kwargs)


class Damage:
    """ 
        Compute the damage introduced both in categorical and numerical attributes 
         Compares a transformed dataset with its original    
    """

    # attribute, type=num,cat, damage,

    def __init__(self, numerical_fn=None):
        """ Compute the damage between the two datasets.
        :param numerical_fn: function used to compute damage for numerical attributes """
        self.numerical_fn = numerical_fn if numerical_fn is not None else self.relative_change

    def relative_change(self, original: pd.core.frame.DataFrame, transformed: pd.core.frame.DataFrame):
        """ Compute the relative change between the two given sets.
        Formula: |original - transformed| / f(original, transformed); with f(o,t) = 1/2 * (|o| + |t|) """

        return (original - transformed).abs() / ((original.abs() + transformed.abs()) / 2)

    def damage_categorical(self, cat_orig, cat_transformed):
        """
        Compute the damage on categorical attributes
        :param cat_orig: categorical attributes. Original
        :param cat_transformed: categorical attributes. Transformed version
        :return: the damage computed
        """
        damage = {}
        for c in cat_orig:
            damage.update({c: [(cat_orig[c] == cat_transformed[c]).mean()]})
        return damage

    def damage_numerical(self, num_orig: pd.core.frame.DataFrame, num_transformed: pd.core.frame.DataFrame):
        """
        Compute the damage on numerical attributes
        :param num_orig: numerical attributes original
        :param num_transformed: numerical attributes transformed
        :return: the damage in form of dataFrame.
        """
        return self.numerical_fn(num_orig, num_transformed)

    def __call__(self, original, transformed):
        if isinstance(original, d.Encoder):
            original = original.df
        if isinstance(transformed, d.Encoder):
            transformed = transformed.df

        cat_clm = original.select_dtypes(exclude=["int", "float", "double"]).columns.tolist()
        num_clm = original.select_dtypes(include=["int", "float", "double"]).columns.tolist()
        d_cat = self.damage_categorical(original[cat_clm], transformed[cat_clm])
        d_num = self.damage_numerical(original[num_clm], transformed[num_clm])

        return d_cat, d_num


class DimensionalityReduction(Visualisation.Visualize):
    """ Use umap to visualize both the original and the transformed dataset """

    def __init__(self, seed=42):
        super().__init__(seed)


def heuristic_single_alpha(results, attribute, optimal_point={"Ber": 0.5, "Fidelity": 1}, max_=False):
    """ Take a result file as input and return the selected epoch """
    data = results[results["Attribute"] == attribute]
    if max_:
        data = data.loc[data.groupby("epoch")["Ber"].idxmax()]
    else:
        data = data.loc[data.groupby("epoch")["Ber"].idxmin()]
    data = data[data["epoch"] != 0]
    data["H"] = (data["Ber"] - optimal_point["Ber"]) ** 2 + (data["Fidelity"] - optimal_point["Fidelity"]) ** 2
    return data[data["H"] == data["H"].min()].iloc[0]["epoch"]
