# Author: Zhengying Liu 
# Creation date: 12 Aug 2021

from mlt.data import DAMatrix

from mlt.metric import AccuracyMetric
from mlt.metric import ArgmaxMeanMetric
from mlt.metric import EmpArgmaxMetric
from mlt.metric import AverageRankMetric

import numpy as np

### Test cases ###
perfs = np.array([
        [0, 1],
        [0, 1],
    ])
da_te = DAMatrix(perfs=perfs)
da_te.best_algo = 1
dist_pred = np.array(
    [0.4, 0.6]
)


def test_AccuracyMetric():
    accuray_metric = AccuracyMetric()
    assert accuray_metric(dist_pred, da_te) == 0.6


def test_ArgmaxMeanMetric():
    argmax_mean_metric = ArgmaxMeanMetric()
    assert argmax_mean_metric(dist_pred, da_te) == 0.6


def test_EmpArgmaxMetric():
    emp_argmax_metric = EmpArgmaxMetric()
    assert emp_argmax_metric(dist_pred, da_te) == 0.6


def test_AverageRankMetric():
    average_rank_metric = AverageRankMetric()
    assert average_rank_metric(dist_pred, da_te) == 0.2


if __name__ == '__main__':
    test_AccuracyMetric()
    test_ArgmaxMeanMetric()
    test_EmpArgmaxMetric()
    test_AverageRankMetric()