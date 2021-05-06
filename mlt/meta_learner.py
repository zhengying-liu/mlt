# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from typing import List

import json
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

from mlt.data import DAMatrix
from mlt.data import CopulaCliqueDAMatrix
from mlt.utils import save_fig


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
        self.name = 'random'
        self.max_print = 3
        self.n_print = 0

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        n_algos = len(da_matrix.algos)

        # Random order of algos for random search
        indices_algo_to_reveal = np.random.permutation(n_algos)
        if self.n_print < self.max_print:
            print("Random search indices_algo_to_reveal", 
                    indices_algo_to_reveal)
            self.n_print += 1
        for i_algo in indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


class OnceRandomSearchMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        """Nothing to do for random search"""
        self.name = 'random'
        n_algos = len(da_matrix.algos)

        # Random order of algos for random search
        self.indices_algo_to_reveal = np.random.permutation(n_algos)
        print("Once random search indices_algo_to_reveal", 
              self.indices_algo_to_reveal)

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
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
        if filtered_da_matrix.shape[0] == 0:
            print("Warning: there are no rows in the DA matrix. " +
                  "Will make random predictions.")
        self.theta_estimation = np.mean(filtered_da_matrix, axis=0)
        self.indices_algo_to_reveal = np.array(np.argsort(self.theta_estimation)[::-1])
        # print("Mean indices_algo_to_reveal", self.indices_algo_to_reveal)
        if da_matrix.algos[0] != "Algorithm 0":
            algos_to_reveal = [da_matrix.algos[i] for i in self.indices_algo_to_reveal]
            # print("Mean algorithms to reveal:", algos_to_reveal)

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


class GreedyMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'greedy'

        n_algos = len(da_matrix.algos)

        # Exlude the indices for validation
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

            # Only let the rows with row[idx] == 0 remain
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
        
        self.indices_algo_to_reveal = np.array(indices_algo_to_reveal)

        print("Greedy indices_algo_to_reveal", self.indices_algo_to_reveal)


    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


class GreedyPlusMetaLearner(S0A1MetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'greedy+'

        n_algos = len(da_matrix.algos)

        # Exlude the indices for validation
        excluded_indices = set(excluded_indices)
        filtered_da_matrix = []
        for i, row in enumerate(da_matrix.perfs):
            if not i in excluded_indices:
                filtered_da_matrix.append(row)
        filtered_da_matrix = np.array(filtered_da_matrix)

        indices_algo_to_reveal = []
        # Try all algorithms in the first step and compare LC of the second idx
        max_alc = 0
        mean_remaining = np.mean(filtered_da_matrix, axis=0)
        for i1 in range(n_algos):
            cond_prob = get_conditional_prob(filtered_da_matrix, cond_cols=[i1])
            i2 = np.argmax(cond_prob)
            alc = mean_remaining[i1] + cond_prob[i2]
            # print("i1, i2, mean_remaining[i1], cond_prob[i2], alc", i1, i2, mean_remaining[i1], cond_prob[i2], alc)
            if alc > max_alc:
                indices_algo_to_reveal = [i1, i2]
                max_alc = alc
        
        i1, i2 = indices_algo_to_reveal
        new_filtered_da_matrix = []
        for row in filtered_da_matrix:
            if row[i1] == 0 and row[i2] == 0:
                new_filtered_da_matrix.append(row)
        new_filtered_da_matrix = np.array(new_filtered_da_matrix)
        filtered_da_matrix = new_filtered_da_matrix

        while len(filtered_da_matrix) > 0 and\
              len(indices_algo_to_reveal) < n_algos:
            se = set(indices_algo_to_reveal)
            indices_remaining = [i for i in range(n_algos) if not i in se]
            mean_remaining = np.mean(filtered_da_matrix, axis=0)
            cols_remaining = mean_remaining[indices_remaining]
            idx = indices_remaining[np.argmax(cols_remaining)]
            indices_algo_to_reveal.append(idx)

            # Only let the rows with row[idx] == 0 remain
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
        
        self.indices_algo_to_reveal = np.array(indices_algo_to_reveal)

        print("Greedy+ indices_algo_to_reveal", self.indices_algo_to_reveal)


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

        self.indices_algo_to_reveal = np.arange(n_algos)

        for perm in all_perms(elements):
            alc = get_meta_train_alc(perm, perfs_meta_train)
            if alc > max_alc:
                self.indices_algo_to_reveal = perm
                max_alc = alc
        self.indices_algo_to_reveal = np.array(self.indices_algo_to_reveal)
        print("Optimal indices_algo_to_reveal", self.indices_algo_to_reveal)

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))


def get_conditional_prob(perfs, i_target=None, cond_cols=None, cond_value=0):
    if cond_cols is None:
        cond_cols = []
    
    for col in cond_cols:
        perfs = perfs[perfs[:, col] == cond_value]
    
    cond_prob = np.mean(perfs, axis=0)
    if i_target is None:
        return cond_prob
    else:
        return cond_prob[i_target]


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
                            excluded_indices=None,
                            show_title=False):
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

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        alc = sum(mean_perfs) / len(mean_perfs)
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
                    label="{} - {}".format(meta_learner.name, alc),
                    transform=trans_offset,
                    marker='o',
                    markersize=5,
                    )

    plt.xlabel("# algorithms tried so far")
    plt.ylabel('proba at least one algo succeeded')
    title = "Learning curve on \nda-matrix: {}, n_runs: {}"\
        .format(da_matrix.name, n_runs)
    if show_title:
        plt.title(title)
    plt.legend(loc='best')
    plt.show()


def run_leave_one_out(meta_learners, da_matrix, n_runs=100, fig=None, 
                      use_all=False, show_title=False):
    """
    Args: 
      meta_learners: list of S0A1MetaLearner objects
      da_matrix: DAMatrix object
      n_runs: int, number of leave-one-out runs
    """
    if fig is None:
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
    else:
        ax = fig.axes[0]

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    n_datasets = len(da_matrix.perfs)
    if n_runs > n_datasets:
        n_runs = n_datasets

    indices_dataset = np.random.choice(n_datasets, n_runs)
    if use_all:
        indices_dataset = range(n_datasets)

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
                    color=get_meta_learner_color(meta_learner.name),
                    barsabove=True,
                    capsize=2,
                    label=meta_learner.name,
                    transform=trans_offset,
                    marker='o',
                    markersize=5,
                    )

    plt.xlabel("# algorithms tried so far")
    plt.ylabel('proba at least one algo succeeded')
    title = "Learning curve on \nda-matrix: {}, n_runs: {}"\
        .format(da_matrix.name, n_runs)
    if show_title:
        plt.title(title)
    plt.legend(loc='best')
    plt.show()

    return fig


def get_meta_learner_color(meta_learner_name):
    if meta_learner_name == 'mean':
        color = 'green'
    elif meta_learner_name == 'greedy':
        color = 'red'
    elif meta_learner_name == 'optimal':
        color = 'orange'
    elif meta_learner_name == 'greedy+':
        color = 'darkred'
    else:
        color = None
    return color


def get_meta_learner_marker(meta_learner_name):
    if meta_learner_name == 'mean':
        marker = '<'
    elif meta_learner_name == 'greedy':
        marker = 'o'
    elif meta_learner_name == 'greedy+':
        marker = '+'
    elif meta_learner_name in {'random', 'random_search'}:
        marker = 's'
    elif meta_learner_name == 'optimal':
        marker = '*'
    else:
        marker = None
    return marker


def get_markevery(n_points, n_markers=10):
    if n_points < 30:
        return 1
    else:
        return n_points // n_markers


def get_markersize(meta_learner_name):
    if meta_learner_name == 'optimal':
        return 5
    else:
        return 10


def run_once_random(da_matrix, perc_valid=0.5, n_meta_learners=100, fig=None,
                    show_legend=True, show_fig=False, leave_one_out=False,
                    show_title=False, figsize=(5,3)):
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)

    n_datasets = len(da_matrix.perfs)
    n_algos = len(da_matrix.algos)

    if leave_one_out:
        n_meta_learners = n_datasets

    learning_curves = []
    for i in range(n_meta_learners):
        meta_learner = OnceRandomSearchMetaLearner()

        if not leave_one_out:
            n_valid = int(perc_valid * n_datasets) # Number of lines for meta-validation
            indices_valid = range(n_datasets - n_valid, n_datasets)
        else:
            indices_valid = [i]

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
    
        learning_curves.append(mean_perfs)

    learning_curves = np.array(learning_curves)

    median_perfs = np.quantile(learning_curves, q=0.5, axis=0)

    cmap = plt.get_cmap('Purples')

    q_perfss = []
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for i, quantile in enumerate(quantiles):
        q_perfs = np.quantile(learning_curves, q=quantile, axis=0)
        q_perfss.append(q_perfs)
        if len(q_perfss) >= 2:
            x = np.arange(n_algos) + 1
            y1 = q_perfss[-2]
            y2 = q_perfss[-1]
            qp1 = int(quantiles[i - 1] * 100)
            qp2 = int(quantiles[i] * 100)
            label = "Quantiles {}-{}%".format(qp1, qp2)
            color = cmap(quantile)
            ax.fill_between(x, y1, y2, label=label, color=color)

    alc = sum(median_perfs) / len(median_perfs)

    ax.plot(np.arange(n_algos) + 1, 
                    median_perfs, 
                    label="{} - {:.4f}".format(meta_learner.name, alc),
                    # transform=trans_offset,
                    marker=get_meta_learner_marker(meta_learner.name),
                    markersize=get_markersize(meta_learner.name),
                    markevery=get_markevery(n_algos),
                    )
    
    plt.xlabel("# algorithms tried so far")
    plt.ylabel('proba at least one algo succeeded')
    title = "Learning curve on \nda-matrix: {}"\
        .format(da_matrix.name)
    if show_title:
        plt.title(title)
    if show_legend:
        plt.legend(loc='best')
    if show_fig:
        plt.show()

    return fig


def run_meta_validation(meta_learners, da_matrix, perc_valid=0.5, fig=None,
                        with_error_bars=False, show_title=False, ylim=None,
                        show_legend=True, figsize=(5,3), show_alc=True,
                        shuffle_row=True):
    """Run meta-training on and meta-validation by making a train/valid split.

    Args: 
      meta_learners: list of S0A1MetaLearner objects
      da_matrix: DAMatrix object
      perc_valid: float, percentage for validation. Will use **last** examples
        for validation.
      fig: plt.figure 
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    else:
        ax = fig.axes[0]

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    n_datasets = len(da_matrix.perfs)
    n_algos = len(da_matrix.algos)

    perfs = da_matrix.perfs
    if shuffle_row:
            permutation = np.random.permutation(n_datasets)
            idx = np.empty_like(permutation)
            idx[permutation] = np.arange(len(permutation))
            perfs[:] = perfs[idx, :]
            da_matrix.perfs = perfs  #TODO: avoid in place changes

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
        alc = sum(mean_perfs) / len(mean_perfs)
        std_perfs = np.std(perfs_arr, axis=0)

        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, 
                                               x=1.5*im, 
                                               y=-1.5*im, 
                                               units='points')

        # Error bar for estimating the mean performance
        yerr = std_perfs / np.sqrt(n_valid)

        if show_alc:
            label = "{} - {:.4f}".format(meta_learner.name, alc)
        else:
            label = meta_learner.name
        
        # Plotting learning curves with error bars
        if with_error_bars:
            ax.errorbar(np.arange(len(mean_perfs)) + 1, 
                        mean_perfs, 
                        yerr=yerr,
                        color=get_meta_learner_color(meta_learner.name),
                        barsabove=True,
                        capsize=2,
                        label=label,
                        transform=trans_offset,
                        marker=get_meta_learner_marker(meta_learner.name),
                        markersize=get_markersize(meta_learner.name),
                        markevery=get_markevery(len(mean_perfs)),
                        alpha=0.5,
                        )
        else:
            ax.plot(np.arange(len(mean_perfs)) + 1, 
                        mean_perfs, 
                        color=get_meta_learner_color(meta_learner.name),
                        label=label,
                        # transform=trans_offset,
                        marker=get_meta_learner_marker(meta_learner.name),
                        markersize=get_markersize(meta_learner.name),
                        markevery=get_markevery(len(mean_perfs)),
                        alpha=0.5,
                        )
        ax.set_xlabel("# algorithms tried so far")
    
    if n_algos <= 10:
        ticks = (np.arange(n_algos) + 1).astype(int)
        labels = [str(x) for x in ticks]
        plt.xticks(ticks=ticks, labels=labels)
    plt.ylabel('proba at least one algo succeeded')
    title = "Learning curve on \nda-matrix: {}"\
        .format(da_matrix.name)
    if show_title:
        plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    if show_legend:
        if da_matrix.name == 'OpenML-Alors':
            plt.legend(loc='lower right')
            plt.xscale('log')
        else:
            plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return fig


def plot_multiple_da_matrices(meta_learner, da_matrices, perc_valid=0.5, 
                              with_error_bars=False, fixed_ylim=False, 
                              show_title=False, return_alc_error_bar=False,
                              figsize=(5,3)):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    alcs = []
    alc_error_bars = []

    max_n_algos = 0

    for im, da_matrix in enumerate(da_matrices):
        n_datasets = len(da_matrix.perfs)
        n_algos = len(da_matrix.algos)
        max_n_algos = max(max_n_algos, n_algos)

        n_valid = int(perc_valid * n_datasets) # Number of lines for meta-validation
        indices_valid = range(n_datasets - n_valid, n_datasets)

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
        alc = sum(mean_perfs) / len(mean_perfs)
        std_alc = np.std(np.mean(perfs_arr, axis=1))
        std_perfs = np.std(perfs_arr, axis=0)

        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, 
                                            x=1.5*im, 
                                            y=-1.5*im, 
                                            units='points')

        # Error bar for estimating the mean performance
        yerr = std_perfs / np.sqrt(n_valid)

        if with_error_bars:
            # Plotting learning curves with error bars
            ax.errorbar(np.arange(len(mean_perfs)) + 1, 
                        mean_perfs, 
                        yerr=yerr,
                        barsabove=True,
                        capsize=2,
                        label="{} - {:.4f}".format(da_matrix.name, alc),
                        transform=trans_offset,
                        marker=get_meta_learner_marker(meta_learner.name),
                        markersize=10,
                        markevery=get_markevery(len(mean_perfs)),
                        alpha=0.5,
                        )
        else:
            ax.plot(np.arange(len(mean_perfs)) + 1, 
                        mean_perfs, 
                        label="{} - {:.4f}".format(da_matrix.name, alc),
                        # transform=trans_offset,
                        marker=get_meta_learner_marker(meta_learner.name),
                        markersize=10,
                        markevery=get_markevery(len(mean_perfs)),
                        alpha=0.5,
                        )

        alcs.append(alc)
        alc_error_bars.append(std_alc / np.sqrt(n_valid))

    plt.xlabel("# algorithms tried so far")
    if max_n_algos <= 10:
        ticks = (np.arange(max_n_algos) + 1).astype(int)
        labels = [str(x) for x in ticks]
        plt.xticks(ticks=ticks, labels=labels)
    plt.ylabel('proba at least one algo succeeded')
    if fixed_ylim:
        plt.ylim(0.45, 1.05)
    title = "Learning curve with \nmeta-learner: {}"\
        .format(meta_learner.name)
    if show_title:
        plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    if return_alc_error_bar:
        return fig, alcs, alc_error_bars
    else:
        return fig, alcs


def binarize(matrix, quantile=0.5):
    threshold = np.quantile(matrix, quantile)
    binarized_matrix = (matrix > threshold).astype(int)
    return binarized_matrix


def save_perfs(perfs, name_expe=None, results_dir='../results', 
               filename=None):
    if filename is None:
        if name_expe is None:
            filename = 'perfs.npy'
        else:
            filename = '{}-perfs.npy'.format(name_expe)
    # Create directory for the experiment
    expe_dir = os.path.join(results_dir, str(name_expe))
    os.makedirs(expe_dir, exist_ok=True)
    # Save numpy matrix
    perfs_path = os.path.join(expe_dir, filename)
    np.savetxt(perfs_path, perfs, fmt='%i')


def get_the_meta_learners(exclude_optimal=False, exclude_greedy_plus=True):
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    greedy_meta_learner = GreedyMetaLearner()
    meta_learners = [
        rs_meta_learner, 
        mean_meta_learner, 
        greedy_meta_learner, 
        ]
    if not exclude_greedy_plus:
        greedy_plus_meta_learner = GreedyPlusMetaLearner()
        meta_learners.append(greedy_plus_meta_learner)
    if not exclude_optimal:
        optimal_meta_learner = OptimalMetaLearner()
        meta_learners.append(optimal_meta_learner)
    return meta_learners


def generate_binary_matrix_with_rank(rank, m, n):
    """Generate a binary matrix of shape `(m, n)` of rank `rank`."""
    if rank < 0:
        raise ValueError("The rank should be positive.")
    if rank == 0:
        return np.zeros((m, n))
    if rank > min(m, n):
        raise ValueError("The rank must be smaller than min(m, n).")

    if m <= n:
        while True:
            rmatrix = (np.random.rand(rank, n) < 0.5).astype(int)
            if np.linalg.matrix_rank(rmatrix) == rank:
                break
        if rank < m:
            remaining_rows = []
            for _ in range(m - rank):
                idx = np.random.randint(rank)
                row = rmatrix[idx]
                remaining_rows.append(row)
            remaining_rows = np.concatenate(remaining_rows).reshape(m - rank, n)
            matrix = np.concatenate([rmatrix, remaining_rows])
        else:
            matrix = rmatrix
        return matrix
    else:
        assert rank <= n
        matrix = generate_binary_matrix_with_rank(rank, n, m)
        return matrix.T


def plot_meta_learner_with_different_ranks(n_datasets=20000, n_algos=5):
    da_matrices = []
    for rank in range(1, n_algos + 1):
        U = np.random.rand(n_datasets, rank)
        V = np.random.rand(rank, n_algos)
        matrix = U.dot(V)
        bm = binarize(matrix)

        real_rank = np.linalg.matrix_rank(bm)

        da_matrix = DAMatrix(
            perfs=bm,
            name="Rank: {} - Real rank: {}"\
                .format(rank, real_rank),
        )
        da_matrices.append(da_matrix)
    
    meta_learners = get_the_meta_learners()
    for meta_learner in meta_learners:
        fig, _ = plot_multiple_da_matrices(meta_learner, da_matrices, fixed_ylim=True)
        name_expe = '{}-different-ranks'.format(meta_learner.name)
        save_fig(fig, name_expe=name_expe)
        for i, da_matrix in enumerate(da_matrices):
            filename = "perfs-{}.npy".format(i + 1)
            save_perfs(da_matrix.perfs, name_expe=name_expe, filename=filename)


def plot_meta_learner_with_different_true_ranks(n_datasets=20000, n_algos=5):
    da_matrices = []
    for rank in range(1, n_algos + 1):
        matrix = generate_binary_matrix_with_rank(rank, n_datasets, n_algos)

        da_matrix = DAMatrix(
            perfs=matrix,
            name="Rank: {}".format(rank),
        )
        da_matrices.append(da_matrix)

    alcss = {}
    
    meta_learners = get_the_meta_learners()
    for im, meta_learner in enumerate(meta_learners):
        fig, alcs = plot_multiple_da_matrices(meta_learner, da_matrices)
        name_expe = '{}-different-true-ranks'.format(meta_learner.name)
        save_fig(fig, name_expe=name_expe)
        for i, da_matrix in enumerate(da_matrices):
            filename = "perfs-rank={}.npy".format(i)
            save_perfs(da_matrix.perfs, name_expe=name_expe, filename=filename)

        
        alcss[meta_learner.name] = alcs

    filepath = '../results/alc-vs-rank.json'
    with open(filepath, 'w') as f:
        json.dump(alcss, f)


def plot_alc_vs_rank(show_title=False):
    filepath = '../results/alc-vs-rank.json'
    with open(filepath, 'r') as f:
        alcss = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    epsilon = 1e-2

    for im, ml in enumerate(alcss):
        alcs = np.array(alcss[ml])

        noise = im * epsilon
        
        ax.plot(np.arange(len(alcs)) + 1 + noise, alcs + noise, 
                label=ml,
                marker=get_meta_learner_marker(ml),
                markersize=5,
                markevery=get_markevery(len(alcs)),
        )

    plt.xlabel("Rank of the DA matrix")
    plt.ylabel('Area under Learning Curve (ALC)')
    title = "ALC vs rank"
    if show_title:
        plt.title(title)
    plt.legend(loc='best')
    plt.show()
    name_expe = 'alc-vs-rank'
    save_fig(fig, name_expe=name_expe)
    return fig


def plot_meta_learner_with_different_cardinal_clique(
        n_datasets=20000, 
        n_algos=5):
    da_matrices = []
    for card in range(2, n_algos + 1):
        da_matrix = CopulaCliqueDAMatrix(cardinal_clique=card, 
                        name="Clique cardinal {}".format(card))
        da_matrices.append(da_matrix)
        print("marginal:", np.mean(da_matrix.perfs, axis=0))

    alcss = {}
    
    meta_learners = get_the_meta_learners(exclude_greedy_plus=False)
    for im, meta_learner in enumerate(meta_learners):
        fig, alcs, alc_error_bars = plot_multiple_da_matrices(
            meta_learner, da_matrices, 
            return_alc_error_bar=True)
        name_expe = '{}-different-cardinal-clique'.format(meta_learner.name)
        save_fig(fig, name_expe=name_expe)
        for i, da_matrix in enumerate(da_matrices):
            filename = "perfs-cardinal-clique={}.npy".format(i + 1)
            save_perfs(da_matrix.perfs, name_expe=name_expe, filename=filename)
        
        alcss[meta_learner.name] = [alcs, alc_error_bars]

    filepath = '../results/alc-vs-cardinal-clique.json'
    with open(filepath, 'w') as f:
        json.dump(alcss, f)


def plot_alc_vs_cardinal_clique(with_noise=False, show_title=False, 
                                with_error_bars=True, figsize=(5,3)):
    filepath = '../results/alc-vs-cardinal-clique.json'
    with open(filepath, 'r') as f:
        alcss = json.load(f)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    epsilon = 1e-2

    for im, ml in enumerate(alcss):
        alcs, alc_error_bars = np.array(alcss[ml])

        noise = im * epsilon if with_noise else 0

        if with_error_bars:
            ax.errorbar(np.arange(len(alcs)) + 2 + noise, alcs + noise, 
                    yerr=alc_error_bars, 
                    barsabove=True,
                    capsize=1,
                    label=ml,
                    marker=get_meta_learner_marker(ml),
                    markersize=5,
                    markevery=get_markevery(len(alcs)),
                    color=get_meta_learner_color(ml),
            )
        else:
            ax.plot(np.arange(len(alcs)) + 2 + noise, alcs + noise, 
                    label=ml,
                    marker=get_meta_learner_marker(ml),
                    markersize=10,
                    markevery=get_markevery(len(alcs)),
                    color=get_meta_learner_color(ml),
            )

    plt.xlabel("Cardinal of the minimal clique")
    plt.ylabel('Area under Learning Curve (ALC)')
    title = "ALC vs Clique cardinal"
    if show_title:
        plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    name_expe = 'alc-vs-cardinal-clique'
    save_fig(fig, name_expe=name_expe)
    return fig
    

# For NeurIPS 2021
def plot_error_bar_vs_B(n_T=10, n_B=20, delta=0.05):
    def get_error_bar(n_T, n_B, delta):
        error_bar = np.sqrt((np.log(n_B) + np.log(2 / delta)) / (2 * n_T))
        return error_bar
    error_bars = [get_error_bar(n_T, b, delta) for b in range(1, n_B + 1)]
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.plot(error_bars, markersize=10, marker='o')
    plt.title("Error bar vs |B|")
    plt.xlabel("|B|")
    plt.ylabel("Error bar")

    return fig


def get_meta_scores_vs_n_tasks(da_matrix, meta_learner, 
                               n_meta_train=5,
                               repeat=100):
    """Get meta-scores (meta-train, meta-valid, meta-test) vs number of tasks
    in the meta-training set. This gives a sort of (meta-)learning curves.

    Suppose there are in total `T` tasks in meta-train. At step `t`, choose 
    randomly `t` tasks among the `T` tasks and apply the meta-learner. Use 
    the `T - t` tasks for meta-validation and use meta-test for test score. 
    Repeat this process `repeat` times and compute the mean and std.

    Here we only use the first algorithm predicted by the meta-learner.

    N.B. For a DA matrix, we suppose the first `n_meta_train` tasks are used 
    as meta-train and the rest is used as  meta-test.
    """
    n_datasets = len(da_matrix.datasets)
    if n_meta_train > n_datasets:
        raise ValueError("The number of meta-train tasks should be less than " +
                         "or equal to the total number of tasks." +
                         "But got {} > {}.".format(n_meta_train, n_datasets))
    T = n_meta_train

    # Form meta-training set
    perfs = da_matrix.perfs[:T, :]
    datasets = da_matrix.datasets[:T]
    algos = da_matrix.algos
    name = da_matrix.name + "-meta-train"
    da_meta_train = DAMatrix(perfs=perfs, datasets=datasets, 
                             algos=algos, name=name)

    # Form meta-test set
    perfs = da_matrix.perfs[T:, :]
    datasets = da_matrix.datasets[T:]
    algos = da_matrix.algos
    name = da_matrix.name + "-meta-test"
    da_meta_test = DAMatrix(perfs=perfs, datasets=datasets, 
                            algos=algos, name=name)

    mean_tr = []
    std_tr = []
    mean_va = []
    std_va = []
    mean_te = []
    std_te = []

    for t in range(1, T + 1):
        s_tr = []
        s_va = []
        s_te = []
        for _ in range(repeat):
            # Choose t among T tasks for meta-train, without replacement
            valid_indices = set(np.random.choice(T, T - t, replace=False))
            meta_learner.meta_fit(da_meta_train, valid_indices)
            i_algo = meta_learner.indices_algo_to_reveal[0]
            # print(da_meta_train.algos[i_algo])
            # print(valid_indices)

            # Meta-train & meta-valid score
            sum_tr = 0
            sum_va = 0
            for i in range(T):
                if i in valid_indices:
                    sum_va += da_meta_train.perfs[i, i_algo]
                else:
                    sum_tr += da_meta_train.perfs[i, i_algo]
            avg_tr = sum_tr / t
            avg_va = sum_va / (T - t) if T > t else np.nan
            s_tr.append(avg_tr)
            s_va.append(avg_va)

            # Meta-test score
            avg_te = np.mean(da_meta_test.perfs[:, i_algo])
            s_te.append(avg_te)
        
        mean_tr.append(np.mean(s_tr))
        std_tr.append(np.std(s_tr))
        mean_va.append(np.mean(s_va))
        std_va.append(np.std(s_va))
        mean_te.append(np.mean(s_te))
        std_te.append(np.std(s_te))

    return mean_tr, std_tr, mean_va, std_va, mean_te, std_te
    

def plot_curve_with_error_bars(li_mean, li_std, fig=None, label=None):
    if fig is None:
        fig = plt.figure()

    if len(fig.axes) > 0:
        ax = fig.axes[0]
    else:
        ax = fig.add_subplot(1, 1, 1)

    # Integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    a_mean = np.array(li_mean)
    a_std = np.array(li_std)
    upper = a_mean + a_std
    lower = a_mean - a_std

    X = np.arange(len(li_mean)) + 1
    
    ax.plot(X, li_mean, marker='o', label=label)

    ax.fill_between(X, upper, lower, alpha=0.3)
        
    return fig

    
