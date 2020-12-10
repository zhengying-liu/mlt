# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from typing import List

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from mlt.data import DAMatrix


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
        """Nothing to do for random search"""
        self.name = 'random_search'
        pass

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        n_algos = len(da_matrix.algos)

        # Random order of algos for random search
        indices_algo_to_reveal = np.random.permutation(n_algos)
        # print("Random search indices_algo_to_reveal", indices_algo_to_reveal)
        for i_algo in indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


class MeanMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'mean'

        excluded_indices = set(excluded_indices)
        filtered_da_matrix = []
        for i, row in enumerate(da_matrix.perfs):
            if not i in excluded_indices:
                filtered_da_matrix.append(row)
        filtered_da_matrix = np.array(filtered_da_matrix)
        self.theta_estimation = np.mean(filtered_da_matrix, axis=0)

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        self.indices_algo_to_reveal = np.argsort(self.theta_estimation)[::-1]
        # print(self.theta_estimation)
        # print(self.indices_algo_to_reveal)
        # print("Mean indices_algo_to_reveal", self.indices_algo_to_reveal)

        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


class GreedyMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'greedy'

        n_algos = len(da_matrix.algos)

        excluded_indices = set(excluded_indices)
        filtered_da_matrix = []
        for i, row in enumerate(da_matrix.perfs):
            if not i in excluded_indices:
                filtered_da_matrix.append(row)
        filtered_da_matrix = np.array(filtered_da_matrix)

        indices_algo_to_reveal = []
        while len(filtered_da_matrix) > 0 and\
              len(indices_algo_to_reveal) < n_algos:
            se = set(indices_algo_to_reveal)
            indices_remaining = [i for i in range(n_algos) if not i in se]
            mean_remaining = np.mean(filtered_da_matrix, axis=0)
            cols_remaining = mean_remaining[indices_remaining]
            idx = indices_remaining[np.argmax(cols_remaining)]
            indices_algo_to_reveal.append(idx)
            new_filtered_da_matrix = []
            for row in filtered_da_matrix:
                if row[idx] == 0:
                    new_filtered_da_matrix.append(row)
            new_filtered_da_matrix = np.array(new_filtered_da_matrix)
            filtered_da_matrix = new_filtered_da_matrix
            # print("filtered_da_matrix.shape", filtered_da_matrix.shape)

        if len(indices_algo_to_reveal) < n_algos:
            se = set(indices_algo_to_reveal)
            indices_remaining = [i for i in range(n_algos) if not i in se]
            perm = np.random.permutation(len(indices_remaining))
            indices_remaining = [indices_remaining[perm[i]] 
                                 for i in range(len(indices_remaining))]
            indices_algo_to_reveal += indices_remaining
            assert len(indices_algo_to_reveal) == n_algos
        
        self.indices_algo_to_reveal = indices_algo_to_reveal

        # print("Greedy indices_algo_to_reveal", self.indices_algo_to_reveal)


    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


class OptimalMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'optimal'

        n_algos = len(da_matrix.algos)
        perfs_meta_train = da_matrix.perfs

        elements = list(range(n_algos))

        max_alc = 0

        for perm in all_perms(elements):
            alc = get_meta_train_alc(perm, perfs_meta_train)
            if alc > max_alc:
                self.indices_algo_to_reveal = perm
                max_alc = alc

        # print("Optimal indices_algo_to_reveal", self.indices_algo_to_reveal)

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                yield perm[:i] + elements[0:1] + perm[i:]


def get_meta_train_alc(perm, perfs_meta_train):
    LC = []
    n_datasets = len(perfs_meta_train)
    for idx in perm:
        if len(perfs_meta_train) > 0:
            

            # Update perfs_meta_train by 
            new_perfs_meta_train = []
            for row in perfs_meta_train:
                if row[idx] == 0:
                    new_perfs_meta_train.append(row)
            new_perfs_meta_train = np.array(new_perfs_meta_train)
            perfs_meta_train = new_perfs_meta_train

            lc = 1 - len(perfs_meta_train) / n_datasets
            LC.append(lc)
        else:
            LC.append(LC[-1])
            assert LC[-1] == 1
    ALC = sum(LC)
    return ALC
    


def run_and_plot_learning_curve(meta_learners, da_matrix, 
                            n_runs=9, 
                            i_dataset=-1,
                            excluded_indices=None):
    """
    Args: 
      meta_learners: list of S0A1MetaLearner objects
      da_matrix: DAMatrix object
      n_runs: number of runs (in case the meta-learner has randomness)
      excluded_indices: list of int, indices of rows excluded for meta-training.
    """
    if excluded_indices is None:
        excluded_indices = [-1]

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)


    for im, meta_learner in enumerate(meta_learners):

        # ax = fig.axes[0]

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

        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, 
                                               x=1.5*im, 
                                               y=-1.5*im, 
                                               units='points')

        # plt.plot(mean_perfs, 'b')
        ax.errorbar(np.arange(len(mean_perfs)) + 1, mean_perfs, yerr=std_perfs, 
                    #  linestyle='dashed',
                    # ecolor='red',
                    barsabove=True,
                    capsize=2,
                    label=meta_learner.name,
                    transform=trans_offset,
                    marker='o',
                    markersize=5,
                    )

    plt.xlabel("# algorithms tried so far")
    plt.ylabel('Probability of having found at least one good algo so far')
    title = "Learning curve on \nda-matrix: {}, n_runs: {}"\
        .format(da_matrix.name, n_runs)
    plt.title(title)
    plt.legend()
    plt.show()


def run_leave_one_out(meta_learners, da_matrix, n_runs=100):
    """
    Args: 
      meta_learners: list of S0A1MetaLearner objects
      da_matrix: DAMatrix object
      n_runs: int, number of leave-one-out runs
    """

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    n_datasets = len(da_matrix.perfs)
    if n_runs > n_datasets:
        n_runs = n_datasets

    indices_dataset = np.random.choice(n_datasets, n_runs)

    for im, meta_learner in enumerate(meta_learners):

        li_history = []
        for i_dataset in indices_dataset:
            meta_learner.history = []
            meta_learner.meta_fit(da_matrix, excluded_indices=[i_dataset])
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

        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, 
                                               x=1.5*im, 
                                               y=-1.5*im, 
                                               units='points')

        # plt.plot(mean_perfs, 'b')
        ax.errorbar(np.arange(len(mean_perfs)) + 1, mean_perfs, yerr=std_perfs, 
                    #  linestyle='dashed',
                    # ecolor='red',
                    barsabove=True,
                    capsize=2,
                    label=meta_learner.name,
                    transform=trans_offset,
                    marker='o',
                    markersize=5,
                    )

    plt.xlabel("# algorithms tried so far")
    plt.ylabel('Probability of having found at least one good algo so far')
    title = "Learning curve on \nda-matrix: {}, n_runs: {}"\
        .format(da_matrix.name, n_runs)
    plt.title(title)
    plt.legend()
    plt.show()

    return fig


def run_meta_validation(meta_learners, da_matrix, perc_valid=0.5):
    """Run meta-training on and meta-validation by making a train/valid split.

    Args: 
      meta_learners: list of S0A1MetaLearner objects
      da_matrix: DAMatrix object
      perc_valid: float, percentage for validation. Will use **last** examples
        for validation.
    """

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    n_datasets = len(da_matrix.perfs)

    n_valid = int(perc_valid * n_datasets) # Number of lines for meta-validation
    indices_valid = range(n_datasets - n_valid, n_datasets)

    for im, meta_learner in enumerate(meta_learners):

        # Meta-training
        meta_learner.meta_fit(da_matrix, excluded_indices=indices_valid)

        # Validation
        li_history = []
        for i_dataset in indices_valid:
            meta_learner.history = []
            meta_learner.fit(da_matrix, i_dataset)
            li_history.append(meta_learner.history)
        
        li_perfs = []
        for history in li_history:
            perfs = [perf for _, _, perf in history]
            cs = np.cumsum(perfs)
            binarized_perfs = (cs >= 1).astype(int)
            li_perfs.append(binarized_perfs)
        
        perfs_arr = np.array(li_perfs)
        mean_perfs = np.mean(perfs_arr, axis=0)
        std_perfs = np.std(perfs_arr, axis=0)

        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, 
                                               x=1.5*im, 
                                               y=-1.5*im, 
                                               units='points')

        # Error bar for estimating the mean performance
        yerr = std_perfs / np.sqrt(n_valid)
        
        # Plotting learning curves with error bars
        ax.errorbar(np.arange(len(mean_perfs)) + 1, 
                    mean_perfs, 
                    yerr=yerr,
                    barsabove=True,
                    capsize=2,
                    label=meta_learner.name,
                    transform=trans_offset,
                    marker='o',
                    markersize=5,
                    )

    plt.xlabel("# algorithms tried so far")
    plt.ylabel('Probability of having found at least one good algo so far')
    title = "Learning curve on \nda-matrix: {}"\
        .format(da_matrix.name)
    plt.title(title)
    plt.legend()
    plt.show()

    return fig
    


    
