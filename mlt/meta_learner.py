# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from typing import List

import json
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

# PyTorch for SGDMetaLearner
import torch
from torch.utils.data import DataLoader

from mlt.data import DAMatrix
from mlt.data import CopulaCliqueDAMatrix
from mlt.data import parse_cepairs_data
from mlt.utils import save_fig
from mlt.utils import get_theoretical_error_bar
from mlt.utils import get_average_rank
from mlt.utils import exclude_indices

from scipy.stats import pearsonr

from datetime import datetime


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

    def meta_fit(self, da_matrix: DAMatrix):
        raise NotImplementedError

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        """Given a dataset with index `i_dataset` in the DA matrix `da_matrix`,
        execute the meta-learning strategy. The meta-learner will reveal the 
        performance of algorithms on datasets step by step.
        """
        raise NotImplementedError

    def rec_algo(self):
        """Recommend one algorithm to try first during meta-test. This should
        be called only after calling `self.meta_fit`.

        Returns:
          int or list of float. int for the index of the recommended algorithm.
            list of float for a distribution on the algorithms.
        """
        raise NotImplementedError



class DefaultFitMetaLearner(S0A1MetaLearner):

    def fit(self, da_matrix: DAMatrix, i_dataset: int):
        for i_algo in self.indices_algo_to_reveal:
            perf = da_matrix.eval(i_dataset, i_algo)
            self.history.append((i_dataset, i_algo, perf))

    def rec_algo(self):
        try:
            return self.indices_algo_to_reveal[0]
        except:
            raise RuntimeError("`self.rec_algo` should be called only after " +
                "calling `self.meta_fit`.")


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


class MeanMetaLearner(DefaultFitMetaLearner):

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None):
        self.name = 'mean'

        if excluded_indices is None:
            excluded_indices = []

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

        # Exclude the indices for validation
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


class MaxAverageRankMetaLearner(DefaultFitMetaLearner):
    """Return the algorithm with best average rank."""

    def meta_fit(self, da_matrix):
        self.name = "max-avg-rank"
        avg_rank = get_average_rank(da_matrix.perfs)
        argsort = avg_rank.argsort()
        self.indices_algo_to_reveal = list(argsort)


# Now we want to create a prior ranking with D
# We then vary the number of candidates in the final phase, choosing the top best in D
# Among those we pick the best candidate in F
# We plot its rank in F (meta-training error) and in G (meta-test error)
def get_ofc(D, F, G, debug_=False):
    ''' Get over-fitting curves as a function of # alogorithms'''
    # G is: the generalization errors, the "true" rank, and algorithm IDs (all identical)
    # Get the final phase error rates 
    sh = D.shape
    m=sh[0]
    Fe =  np.zeros(sh)
    Fe[F] = np.arange(m)
    Fe = Fe.astype(int)
    # Get the final phase scores in the order given by the development phase
    Fes = Fe[D]
    if debug_: 
        print(D)
        print(F)
        print(Fe)
        print(Fes)
    # Get training and generalization errors
    Tr = np.zeros(sh)
    Te = np.zeros(sh)
    for j in np.arange(1,m+1):
        if debug_: print(Fes[0:j])
        Tr[j-1] = np.min(Fes[0:j])
        k = np.argmin(Fes[0:j])
        Te[j-1] = D[k]
        assert D[k] == F[Fes[k]]
        assert Tr[j-1] == Fes[k]
    return Tr, Te


class TopkRankMetaLearner(DefaultFitMetaLearner):
    """The `meta_fit` method of this class may not give a full ranking."""

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None, 
                 gdf_ratios=None, repeat=100, plot=False, n_feedback=None):
        """
        Args:
          gdf_ratios: list of length 0.3, the sum should be equal to 1
          repeat: int, number of repetitions of "cross-validation"
          plot: boolean, if plot a figure
          n_feedback: int, first `n_feedback` examples will be used as 
            "feedback phase" data
        """
        self.name = 'top-k-rank'

        if gdf_ratios is None:
            gdf_ratios = [1 / 3] * 3
        
        gdf_cumsum = np.cumsum(gdf_ratios)

        n_algos = len(da_matrix.algos)
        n_excl = len(excluded_indices) if excluded_indices else 0
        n_datasets = len(da_matrix.datasets) - n_excl

        # Exclude the indices for validation
        if excluded_indices is None:
            excluded_indices = {}
        elif not isinstance(excluded_indices, range):
            excluded_indices = set(excluded_indices)
        perfs = []
        for i, row in enumerate(da_matrix.perfs):
            if not i in excluded_indices:
                perfs.append(row)
        perfs = np.array(perfs)
        

        # Maintenant on doit faire plein de tirage et moyenner les learning curves
        m = n_algos
        num_trials = repeat
        TR = np.zeros((num_trials, m))
        TE = np.zeros((num_trials, m))
        C = np.zeros((num_trials,))
        G = np.arange(m)
        
        for t in range(num_trials):
            G_perfs = []
            D_perfs = []
            F_perfs = []
            for i, row in enumerate(da_matrix.perfs):
                if not i in excluded_indices:
                    x = np.random.rand()
                    if x <= gdf_cumsum[0]:
                        G_perfs.append(row)
                    elif x <= gdf_cumsum[1]:
                        D_perfs.append(row)
                    else:
                        F_perfs.append(row)
            G_perfs = np.array(G_perfs) if len(G_perfs) else perfs[np.random.randint(n_datasets)][None, :]
            D_perfs = np.array(D_perfs) if len(D_perfs) else perfs[np.random.randint(n_datasets)][None, :]
            F_perfs = np.array(F_perfs) if len(F_perfs) else perfs[np.random.randint(n_datasets)][None, :]
            assert G_perfs.shape[-1] == m
            assert D_perfs.shape[-1] == m
            assert F_perfs.shape[-1] == m
            Gas = get_average_rank(G_perfs).argsort()
            Das = get_average_rank(D_perfs).argsort()
            Fas = get_average_rank(F_perfs).argsort()
            # Gas = (-G_perfs).sum(axis=0).argsort()
            # Das = (-D_perfs).sum(axis=0).argsort()
            # Fas = (-F_perfs).sum(axis=0).argsort()
            D = Gas[Das]
            F = Gas[Fas]
            G = np.arange(m)
            c = pearsonr(D, G)[0]
        
            Tr, Te = get_ofc(D, F, G)
            TR[t, :] = Tr
            TE[t, :] = Te
            C[t] = c

            # Fe =  np.zeros(m)
            # Fe[F] = np.arange(m)
            # Fe = Fe.astype(int)
            # # Get the final phase scores in the order given by the development phase
            # Fes = Fe[D]
            # Gas_inv = np.zeros(m)
            # Gas_inv[Gas] = np.arange(m)
            # indices_algo_to_reveal = Gas_inv[Fes[:3].argsort()].astype(int)
            # print([da_matrix.algos[i] for i in indices_algo_to_reveal])
        
        Tr = np.mean(TR, axis=0) / m
        Te = np.mean(TE, axis=0) / m
        Correl = np.mean(C)

        # Determine `k`: only top `k` participants in development phase will be
        # considered
        k = Te.argmin() + 1
        self.k = k

        # Use a part of data as feedback and the rest as final
        # Use all data to estimate G
        if n_feedback is None:
            n_filtered = len(da_matrix.perfs) - len(excluded_indices)
            n_feedback = n_filtered // 2
        feedback_perfs = perfs[:n_feedback]
        final_perfs = perfs[n_feedback:]
        Gas = get_average_rank(perfs).argsort()
        Das = get_average_rank(feedback_perfs).argsort()
        Fas = get_average_rank(final_perfs).argsort()
        D = Gas[Das]
        F = Gas[Fas]
        G = np.arange(m)

        Fe =  np.zeros(m)
        Fe[F] = np.arange(m)
        Fe = Fe.astype(int)
        # Get the final phase scores in the order given by the development phase
        Fes = Fe[D]
        Gas_inv = np.zeros(m)
        Gas_inv[Gas] = np.arange(m)
        self.indices_algo_to_reveal = Gas[Fes[:k].argsort()].astype(int)
        final_phase_score = [np.min(Fes[:i+1]) / m for i in range(k)]

        # Validation
        if len(excluded_indices) > 0:
            excluded_perfs = []
            for i, row in enumerate(da_matrix.perfs):
                if i in excluded_indices:
                    excluded_perfs.append(row)
            excluded_perfs = np.array(excluded_perfs)
            E = (-excluded_perfs).sum(axis=0).argsort()
            E_inv = np.zeros(m)
            E_inv[E] = np.arange(m)
            meta_test_score = E_inv[self.indices_algo_to_reveal[0]] / m

        if plot:
            STr = np.std(TR, axis=0) / m
            STe = np.std(TE, axis=0) / m
            STre = 2*STr/np.sqrt(num_trials)
            STee = 2*STe/np.sqrt(num_trials)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.plot(np.arange(m) + 1, Tr, 'ro')
            plt.plot(np.arange(m) + 1, Tr, 'r-', label = 'Meta-train error')
            plt.fill_between(G + 1, (Tr-STre), (Tr+STre), color='red', alpha=0.1)

            plt.plot(np.arange(m) + 1, Te, 'bo')
            plt.plot(np.arange(m) + 1, Te, 'b-', label = 'Meta-valid error')
            plt.fill_between(G + 1, (Te-STee), (Te+STee), color='blue', alpha=0.1)

            if len(excluded_indices) > 0:
                plt.plot(1, meta_test_score, 'black', marker='+')

            plt.plot(np.arange(len(final_phase_score)) + 1, [x for x in final_phase_score], 'g-', label="Final phase error", marker='o')
            # print(final_phase_score)

            plt.xscale('log')

            plt.gca().legend()
            plt.xlabel('Number of Final phase participants')
            plt.ylabel('Average ranking percentage')
            name = da_matrix.name
            plt.title('{}; Error bar = 2 stderr; <Correl>={:.2f}'.format(name, Correl))
            plt.show()

            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            name_expe = "topk-rank"
            filename = "{}-{}-{}".format(name, name_expe, date_str)
            save_fig(fig, name_expe=name_expe, filename=filename)

            # Theoretical bound and difference
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Meta-train - meta-test
            diff_curve = Te - Tr
            ax.plot(np.arange(m) + 1, diff_curve,
                label='meta-test - meta-train', marker='o', markersize=2)

            # Theoretical bounds
            n_T = n_datasets
            n_B = n_algos
            error_bars_the = [get_theoretical_error_bar(n_T, i + 1, delta=0.05) 
                                for i in range(n_B)]
            ax.plot(np.arange(m) + 1, error_bars_the,
                label='Theoretical error bar', marker='o', markersize=2)

            plt.xscale('log')

            plt.gca().legend()
            plt.xlabel('Number of Algorithms')
            plt.ylabel('Ranking difference in percentage')
            name = da_matrix.name
            plt.title('{} - ranking difference'.format(name))
            plt.show()

            name_expe = "topk-rank-diff"
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = "{}-{}-{}".format(name, name_expe, date_str)
            save_fig(fig, name_expe=name_expe, filename=filename)


class PredefinedKRankMetaLearner(DefaultFitMetaLearner):
    """The `meta_fit` method of this class may not give a full ranking."""

    def get_k(self, da_matrix: DAMatrix):
        raise NotImplementedError

    def meta_fit(self, da_matrix: DAMatrix, excluded_indices: List=None, 
                 plot=False, feedback_size=0.5, shuffling=True):
        """
        Args:
          excluded_indices: list, row indices excluded for validation
          plot: boolean, if plot a figure
          n_feedback: int, first `n_feedback` examples will be used as 
            "feedback phase" data
        """
        self.k = self.get_k(da_matrix)

        n_algos = len(da_matrix.algos)
        n_excl = len(excluded_indices) if excluded_indices else 0
        n_datasets = len(da_matrix.datasets) - n_excl

        # Exclude the indices for validation
        perfs, perfs_valid = exclude_indices(
            da_matrix.perfs, 
            excluded_indices)
        # if excluded_indices is None:
        #     excluded_indices = {}
        # elif not isinstance(excluded_indices, range):
        #     excluded_indices = set(excluded_indices)
        # perfs = []
        # for i, row in enumerate(da_matrix.perfs):
        #     if not i in excluded_indices:
        #         perfs.append(row)
        # perfs = np.array(perfs)
        
        # Set k
        k = self.k

        # Use a part of data as feedback and the rest as final
        # Use all data to estimate G
        da_tr, da_te = DAMatrix(perfs=perfs).train_test_split(
            train_size=feedback_size,
            shuffling=shuffling,
        )
        feedback_perfs = da_tr.perfs
        final_perfs = da_te.perfs

        m = n_algos
        Gas = (-perfs).sum(axis=0).argsort()
        Das = (-feedback_perfs).sum(axis=0).argsort()
        Fas = (-final_perfs).sum(axis=0).argsort()
        D = Gas[Das]
        F = Gas[Fas]

        Fe =  np.zeros(m)
        Fe[F] = np.arange(m)
        Fe = Fe.astype(int)
        # Get the final phase scores in the order given by the development phase
        Fes = Fe[D]
        self.indices_algo_to_reveal = Gas[Fes[:k].argsort()].astype(int)


class FixedKRankMetaLearner(PredefinedKRankMetaLearner):

    def __init__(self, k, history=None, name=None):
        """
        Args:
          k: int, top `k` parcipant to enter final phase
        """
        self.k = k
        if name is None:
            name = "top-{}".format(k)
        super().__init__(history=history, name=name)

    def get_k(self, da_matrix):
        return self.k


class TopPercRankMetaLearner(PredefinedKRankMetaLearner):

    def __init__(self, perc, history=None, name=None):
        """
        Args:
          perc: int between (0, 100], percentage of top participants to enter 
            final phase
        """
        self.perc = perc
        if name is None:
            name = "top-{}-perc".format(perc)
        super().__init__(history=history, name=name)

    def get_k(self, da_matrix):
        n_algos = len(da_matrix.algos)
        k = int(n_algos * self.perc / 100)
        return k


class TopKD(DefaultFitMetaLearner):
    """top-k-d method according to AISTATS 2022 paper.
    """
    k = 2

    def meta_fit(self, da_matrix, feedback_size=0.5, shuffling=False):
        self.name = 'top-k-d'

        # Use a part of data as feedback and the rest as final
        da_tr, da_te = DAMatrix(perfs=da_matrix.perfs).train_test_split(
            train_size=feedback_size,
            shuffling=shuffling,
        )
        feedback_perfs = np.array(da_tr.perfs)
        final_perfs = np.array(da_te.perfs)
        d = np.mean(feedback_perfs, axis=0)
        f = np.mean(final_perfs, axis=0)
        perm = np.argsort(-d)

        # Get k
        k = self.k
        top_k_d = perm[:k]
        w_k = top_k_d[np.argmax(f[top_k_d])]

        self.indices_algo_to_reveal = [w_k]


class SRM(DefaultFitMetaLearner):
    """Structural Risk Minimization MetaLearner according to AISTATS 2022 
    paper.
    """

    def meta_fit(self, da_matrix, feedback_size=0.5, shuffling=False, 
            plot=False):
        self.name = 'SRM'

        # Use a part of data as feedback and the rest as final
        da_tr, da_te = DAMatrix(perfs=da_matrix.perfs).train_test_split(
            train_size=feedback_size,
            shuffling=shuffling,
        )
        feedback_perfs = np.array(da_tr.perfs)
        final_perfs = np.array(da_te.perfs)
        d = np.mean(feedback_perfs, axis=0)
        f = np.mean(final_perfs, axis=0)
        perm_d = np.argsort(-d)
        perm_f = np.argsort(-f)

        top_2_d = perm_d[:2]
        fw2 = f[top_2_d[np.argmax(f[top_2_d])]]
        top_2_f = perm_f[:2]
        dw2 = d[top_2_f[np.argmax(d[top_2_f])]]
        
        m = len(da_matrix.algos)
        fw = np.zeros(m)
        gw = np.zeros(m)
        for k in range(1, m + 1):
            top_k_d = perm_d[:k]
            w_k = top_k_d[np.argmax(f[top_k_d])]
            fw[k - 1] = f[w_k]
            gw[k - 1] = fw[k - 1] + abs(dw2 - fw2) * np.sqrt(k - 1)

        if plot:
            fig = plt.figure()
            plt.plot(gw, label='gw')
            plt.plot(fw, label='fw')
            plt.legend()
            plt.show()
            save_fig(fig, name_expe='srm')

        k_star = np.argmax(gw) + 1
        self.k = k_star
        top_k_d = perm_d[:k_star]
        w_k_star = top_k_d[np.argmax(f[top_k_d])]
        self.indices_algo_to_reveal = [w_k_star]


class CountMaxMetaLearner(DefaultFitMetaLearner):
    
    def meta_fit(self, da_matrix: DAMatrix):
        self.name = 'count-max'
        
        n_algos = len(da_matrix.algos)
        # Count the number of times that each algorithm attains the best
        argmaxs = da_matrix.perfs.argmax(axis=-1)
        counter = np.zeros(n_algos, dtype=int)
        for a in argmaxs:
            counter[a] += 1
        self.dist_emp = counter / counter.sum()
        i_hat = self.dist_emp.argmax()
        self.indices_algo_to_reveal = [i_hat]
        
    def rec_algo(self, use_proba=False):
        if use_proba:
            return self.dist_emp
        else:
            return self.indices_algo_to_reveal[0]


class SGDMetaLearner(DefaultFitMetaLearner):
    
    def meta_fit(self, da_matrix: DAMatrix, lr=1e-1, n_epochs=40, batch_size=5):
        self.name = 'sgd'
        n_algos = len(da_matrix.algos)
        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Weights for dist_pred
        self.logits = torch.zeros(n_algos, requires_grad=True)
        w = self.logits
        
        def loss_fn(X): 
            X = X.to(device)
            # Compute prediction error
            X = torch.as_tensor(X, dtype=torch.float32)
            loss = - torch.softmax(w, dim=-1).dot(X.mean(axis=0))
            if w.grad is not None:
                w.grad.data.zero_()
            loss.backward()
            return loss.data

        def optimize(learning_rate):
            w.data -= learning_rate * w.grad.data

        def train(dataloader):
            size = len(dataloader.dataset)
            for i, X in enumerate(dataloader):
                loss = loss_fn(X)
                optimize(lr)
                if i % 100 == 0:
                    loss, current = loss.item(), i * len(X)
        
        # Meta-training
        train_dataloader = DataLoader(
            da_matrix.perfs, 
            batch_size=batch_size, 
            shuffle=True)
        for t in range(n_epochs):
            train(train_dataloader)
        
    def rec_algo(self):
        dist_pred = torch.softmax(self.logits, dim=-1).detach().numpy()
        return dist_pred
        


######################################################
################# Helper Functions ###################
######################################################

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
    
    error_bars = [get_theoretical_error_bar(n_T, b, delta) for b in range(1, n_B + 1)]
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.plot(error_bars, markersize=10, marker='o')
    plt.title("Error bar vs |B|")
    plt.xlabel("|B|")
    plt.ylabel("Error bar")

    return fig



