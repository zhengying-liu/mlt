# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.data import DAMatrix
from typing import List

import numpy as np
import matplotlib.pyplot as plt

class S0A1MetaLearner(object):
    """Abstract class for S0A1 meta-learner according to Zhengying Liu's PhD 
    thesis, Chapter 7 on meta-learning.
    """

    def __init__(self, history: List=None, name=None):
        """
        Args:
          history: list of tuples of the form (i_dataset, i_algo, perf).
        """
        self.history = history if history is None else []
        self.name = name

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        raise NotImplementedError

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        """Given a dataset with index `i_dataset` in the DA matrix `da_matrix`,
        execute the meta-learning strategy. The meta-learner will reveal the 
        performance of algorithms on datasets step by step.
        """
        raise NotImplementedError


class RandomSearchMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'random_search'
        """Nothing to do for random search"""
        pass

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        n_algos = len(da_matrix.algos)

        # Random order of algos for random search
        indices_algo_to_reveal = np.random.permutation(n_algos)
        for i_algo in indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


def run_and_plot_learning_curve(meta_learner, da_matrix, 
                            n_runs=100, 
                            i_dataset=-1,
                            excluded_indices=None):
    if excluded_indices is None:
        excluded_indices = [-1]

    li_history = []
    for i in range(n_runs):
        meta_learner.history = []
        meta_learner.meta_fit(da_matrix, excluded_indices=excluded_indices)
        meta_learner.fit(da_matrix, i_dataset)
        history = meta_learner.history
        li_history.append(history)
    
    li_perfs = []
    for history in li_history:
        perfs = [perf for _, _, perf in history]
        cs = np.cumsum(perfs)
        binarized_perfs = (cs >= 1).astype(int)
        li_perfs.append(binarized_perfs)
    
    perfs_arr = np.array(li_perfs)
    mean_perfs = np.mean(perfs_arr, axis=0)
    std_perfs = np.std(perfs_arr, axis=0)

    # plt.plot(mean_perfs, 'b')
    plt.errorbar(np.arange(len(mean_perfs)), mean_perfs, yerr=std_perfs, 
                #  linestyle='dashed',
                ecolor='red',
                barsabove=True,
                capsize=2,
                label=meta_learner.name,
                )
    plt.xlabel("# iterations")
    plt.ylabel('Frequence of hitting a good algo so far')
    title = "Learning curve on \nda-matrix: {}, n_runs: {}"\
        .format(da_matrix.name, n_runs)
    plt.title(title)
    plt.legend()
    plt.show()
    


    
