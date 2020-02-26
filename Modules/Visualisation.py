import umap
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import numpy as np
import seaborn as sns
import pandas as pd


# Fix numpy seed
# Fix torch seed

class Visualize:

    def plot(self, fn, *args, **kwargs):
        """ Wrapper to force plotting """
        done = False
        while not done:
            try:
                fn(*args, **kwargs)
                done = True
            except (ValueError) as e:
                print("Plot encounter error: ")
                print(e)
                done = False
                if not self.force:
                    raise ValueError(e)

    def __init__(self, seed=42, force=True):
        self.seed = seed
        self.force = force

    def encode_cat(self, dFrame):
        """
        Transform categorical columns into a binary expansion
        :param dFrame: the dataframe to encode
        :return: The encoded version
        """

        cat_clm = dFrame.select_dtypes(exclude=["int", 'float', 'double']).columns.tolist()
        return pd.get_dummies(dFrame, columns=cat_clm)

    def umap(self, dFrame, min_dist=0.2, n_components=2, plots=False):
        """
        Visualization with umap
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        umap_data = umap.UMAP(min_dist=min_dist, n_components=n_components).fit_transform(dFrame.values)
        if plots:
            self.plots("Decomposition using UMAP", umap_data, n_components)
        return umap_data

    def tsne(self, dFrame, n_components=2, plots=False):
        """
        Visualisation with t-sne
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        tsne = TSNE(n_components=n_components).fit_transform(dFrame.values)
        if plots:
            self.plots("t-SNE components", tsne, n_components)
        return tsne

    def pca(self, dFrame, n_components=2, plots=False):
        """
        Visualisation with pca
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(dFrame.values)
        plt.figure(figsize=(12, 8))
        if plots:
            self.plots("Component-wise and Cumulative Explained Variance", pca_result, n_components)
        return pca_result

    def svd(self, dFrame, n_components=2, plots=False):
        """
        Single value decomposition visualisation
        :param dFrame:
        :return:
        """
        dFrame = self.encode_cat(dFrame)
        svd = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(dFrame.values)
        if plots:
            self.plots("SVD Components", svd, n_components)
        return svd

    def plots(self, title, components, n_components=2):
        plt.figure(figsize=(12, 8))
        plt.title(title)
        if n_components > 2:
            for i in range(n_components):
                self.plot(sns.scatterplot, components[:, i % n_components], components[:, (i + 1) % n_components])
        else:
            self.plot(sns.scatterplot, components[:, 0], components[:, 1])

    def reachability_plot(self, labels, reachability, indexes, savefig=None):
        """
        Plot the reachability of all clusters
        :param labels: the clusters labels
        :param reachability: the indexes reachability
        :param indexes: the dataset indexes
        :return: Nothing
        """
        df = pd.DataFrame.from_dict({"reachability": reachability, "indexes": indexes, "labels": labels})
        plt.figure(figsize=(10, 7))
        self.plot(sns.scatterplot, data=df, x="indexes", y="reachability", hue="labels")

        if savefig is not None:
            plt.savefig(savefig)

    def clusters_plots(self, dFrame, labels, dimRedFn="umap", savefig=None, *args, **kwargs):
        """
        Shows the clusters using dimensionality reduction
        :param dFrame: the original dataFrame (n_samples, n_features)
        :param labels: the clusters label (n_samples)
        :param dimRedFn: dimensionality reduction technique, options are: "umap", "tsne", "pca", "svd"
        should return 2 components
        :param args: dimensionality reduction arguments
        :param kwargs: dimensionality reduction keyword arguments
        """
        if "umap" in dimRedFn.lower():
            components = self.umap(dFrame, *args, **kwargs)
        elif "tsne" in dimRedFn.lower():
            components = self.tsne(dFrame, *args, **kwargs)
        elif "pca" in dimRedFn.lower():
            components = self.pca(dFrame, *args, **kwargs)
        elif "svd" in dimRedFn.lower():
            components = self.svd(dFrame, *args, **kwargs)
        else:
            components = self.dimRedFn(self.encode_cat(dFrame), *args, **kwargs)

        df = pd.DataFrame.from_dict({
            "labels": labels,
            "1st_component": components[:, 0],
            "2nd_component": components[:, 1],
        })
        plt.figure()
        self.plot(sns.scatterplot, data=df, x="1st_component", y="2nd_component", hue="labels")
        if savefig is not None:
            plt.savefig(savefig)

    def original_vs_transformed_plots(self, dFrames, dimRedFn="umap", savefig=None, *args, **kwargs):
        """
        Shows the clusters using dimensionality reduction
        :param dFrames: dictionary of frames to reduce dim and plots
        :param dimRedFn: dimensionality reduction technique, options are: "umap", "tsne", "pca", "svd"
        should return 2 components
        :param args: dimensionality reduction arguments
        :param kwargs: dimensionality reduction keyword arguments
        """
        if "umap" in dimRedFn.lower():
            reduction_fn = self.umap
        elif "tsne" in dimRedFn.lower():
            reduction_fn = self.tsne
        elif "pca" in dimRedFn.lower():
            reduction_fn = self.pca
        elif "svd" in dimRedFn.lower():
            reduction_fn = self.svd
        else:
            reduction_fn = lambda x, *args, **kwargs: self.dimRedFn(self.encode_cat(x), *args, **kwargs)

        components = {}
        df = pd.DataFrame()
        for k, dFrame in dFrames.items():
            r = reduction_fn(dFrame, *args, **kwargs)
            components.update({k: r})
            df = pd.concat([df, pd.DataFrame.from_dict({
                "Type": k,
                "1st_component": r[:, 0],
                "2nd_component": r[:, 1]
            })])

        fig_id = str(np.random.random(20))
        plt.figure(fig_id, figsize=(14, 14))
        self.plot(sns.scatterplot, data=df, x="1st_component", y="2nd_component", hue="Type")
        if savefig is not None:
            plt.savefig(savefig)
        plt.close(fig_id)

    def clusters_original_vs_transformed_plots(self, dFrames, labels, dimRedFn="umap", savefig=None, *args, **kwargs):
        """
        Shows the clusters using dimensionality reduction
        :param dFrames: dictionary of frames to reduce dim and plots
        :param labels: the clusters label (n_samples).
        :param dimRedFn: dimensionality reduction technique, options are: "umap", "tsne", "pca", "svd"
        should return 2 components
        :param args: dimensionality reduction arguments
        :param kwargs: dimensionality reduction keyword arguments
        """
        if "umap" in dimRedFn.lower():
            reduction_fn = self.umap
        elif "tsne" in dimRedFn.lower():
            reduction_fn = self.tsne
        elif "pca" in dimRedFn.lower(): ###
            reduction_fn = self.pca
        elif "svd" in dimRedFn.lower():
            reduction_fn = self.svd
        else:
            reduction_fn = lambda x, *args, **kwargs: self.dimRedFn(self.encode_cat(x), *args, **kwargs)

        components = {}
        df = pd.DataFrame()
        for k, dFrame in dFrames.items():
            r = reduction_fn(dFrame, *args, **kwargs)
            components.update({k: r})
            d = pd.DataFrame.from_dict({
                "T": k,
                "1st_component": r[:, 0],
                "2nd_component": r[:, 1]
            })
            d["Type"] = d["T"] + "-" + labels.astype(str)
            df = pd.concat([df, d])

        fig_id = str(np.random.random(20))
        plt.figure(fig_id, figsize=(14, 14))
        self.plot(sns.scatterplot, data=df, x="1st_component", y="2nd_component", hue="Type")
        if savefig is not None:
            plt.savefig(savefig)
        plt.close(fig_id)
