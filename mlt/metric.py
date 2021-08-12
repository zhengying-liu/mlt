# Author: Zhengying Liu 
# Creation date: 12 Aug 2021

from collections.abc import Iterable
from mlt.utils import get_average_rank

import numpy as np


class MetaTestMetric(object):
    
    def __init__(self, name=None):
        
        self.name = name
        
    def __call__(self, dist_pred, da_te):
        """
        Args:
          dist_pred: list of length `A` (number of algorithms), a probability distribution on {1,...,A}.
          da_te: DAMatrix of shape (D, A) for meta-test, where D is the number of tasks/datasets.
        """
        raise NotImplementedError


class AccuracyMetric(MetaTestMetric):
    """Compute the accuracy when the ground truth is known."""
    
    def __init__(self, name='likelyhood-metric'):
        super().__init__(name=name)

    def __call__(self, dist_pred, da_te):
        """Compute the accuracy when the ground truth is known.

        Args:
          dist_pred: int or list of length `A` (number of algorithms), a 
            probability distribution on {1,...,A}.
          da_te: DAMatrix of shape (D, A) for meta-test, where D is the number 
            of tasks/datasets.
        """
        try:
            i_star = da_te.best_algo
        except:
            raise ValueError("The best algorithm of this DA matrix is unknown!")
        if not isinstance(dist_pred, Iterable):
            if dist_pred == i_star:
                return 1.0
            else:
                return 0.0
        likelyhood = dist_pred[i_star]
        return likelyhood


class ArgmaxMeanMetric(MetaTestMetric):
    """Use the empirical distribution obtained from the DA matrix as an 
    estimation of the true distribution.
    """
    
    def __init__(self, name='argmax-mean-metric'):
        super().__init__(name=name)

    def __call__(self, dist_pred, da_te):
        """Use the empirical distribution obtained from the DA matrix as an 
        estimation of the true distribution.

        Args:
          dist_pred: int or list of length `A` (number of algorithms), a 
            probability distribution on {1,...,A}.
          da_te: DAMatrix of shape (D, A) for meta-test, where D is the number 
            of tasks/datasets.
        """
        perfs = da_te.perfs
        if len(perfs.shape) < 2:
            raise ValueError("`matrix` should be at least 2-D.")
        # Compute empirical argmax / best algo
        mean_perfs = perfs.mean(axis=-2)
        i_hat = mean_perfs.argmax()
        if not isinstance(dist_pred, Iterable):
            if dist_pred == i_hat:
                return 1.0
            else:
                return 0.0
        likelyhood_hat = dist_pred[i_hat]
        return likelyhood_hat


class EmpArgmaxMetric(MetaTestMetric):
    """WARNING: q(A)=0 should imply p(A)=0 
        i.e. p(A)!=0 should imply q(A)!=0
        i.e. p is absolutely continuously w.r.t q).
    """
    
    def __init__(self, name='emp-argmax-metric'):
        super().__init__(name=name)

    def __call__(self, dist_pred, da_te):
        """Use the empirical distribution on argmax (on a given task) as the 
        ground truth (distribution). In this case, the ground truth is no more 
        deterministic. Will compute exponential minus KL divergence.

        Args:
          dist_pred: int or list of length `A` (number of algorithms), a 
            probability distribution on {1,...,A}.
          da_te: DAMatrix of shape (D, A) for meta-test, where D is the number 
            of tasks/datasets.
        """
        perfs = da_te.perfs
        n_algos = perfs.shape[-1]
        if not isinstance(dist_pred, Iterable):
            i = dist_pred
            dist_pred = np.zeros(n_algos)
            dist_pred[i] = 1

        # Obtain empirical distribution of the argmax
        argmaxs = perfs.argmax(axis=-1)
        counter = np.zeros(n_algos, dtype=int)
        for a in argmaxs:
            counter[a] += 1
        dist_emp = counter / counter.sum()

        # Compute exponential of minus KL divergence
        p = dist_emp
        q = dist_pred
        exp_kl_div = 1
        for i in range(n_algos):
            if p[i] != 0:
                if q[i] == 0:
                    return 0
                else:
                    # Exponential of minus KL divergence
                    exp_kl_div *= (p[i] / q[i]) ** (- p[i])  

        return exp_kl_div


class AverageRankMetric(MetaTestMetric):
    
    def __init__(self, name='average-rank-metric'):
        super().__init__(name=name)

    def __call__(self, dist_pred, da_te):
        """Compute the expected average rank of `dist_pred`.

        Args:
          dist_pred: int or list of length `A` (number of algorithms), a 
            probability distribution on {1,...,A}.
          da_te: DAMatrix of shape (D, A) for meta-test, where D is the number 
            of tasks/datasets.
        """
        perfs = da_te.perfs
        n_algos = perfs.shape[-1]
        avg_rank = get_average_rank(perfs, normalized=True)
        res = 0
        # When the predicted distribution is one index
        if not isinstance(dist_pred, Iterable): 
            return avg_rank[dist_pred]
        for i in range(n_algos):
            res += dist_pred[i] * avg_rank[i]
        return res

