import numpy as np
import pandas as pd


class DiversityDamageResults:
    """ Method for storing the diversity and damage results. """

    def __init__(self, resultDir, result_file="DiversityDamage.csv", num_damage="numerical_damage.csv", overwrite=False):
        self.results_dir = resultDir
        self.result_file = result_file
        self.num_damage = num_damage
        self.rd = None
        self.nd = None
        if not overwrite:
            try:
                self.rd = pd.read_csv("{}/{}".format(self.results_dir, self.result_file))
                self.nd = pd.read_csv("{}/{}".format(self.results_dir, self.num_damage))
            except (FileNotFoundError, pd.errors.EmptyDataError):
                self.rd = None
                self.nd = None

    def add_results(self, diversity, damage_categorical, damage_numerical, **params):
        """ """
        all_data = {}
        all_data.update(diversity)
        all_data.update(damage_categorical)
        all_data = pd.DataFrame.from_dict(all_data)

        for k, v in params.items():
            all_data[k] = v
            damage_numerical[k] = v
        if self.rd is None:
            self.rd = all_data
        else:
            self.rd = pd.concat([self.rd, all_data], axis=0)
        if self.nd is None:
            self.nd = damage_numerical
        else:
            self.nd = pd.concat([self.nd, all_data], axis=0)

        self.nd.to_csv("{}/{}".format(self.results_dir, self.num_damage), index=False)
        self.rd.to_csv("{}/{}".format(self.results_dir, self.result_file), index=False)

