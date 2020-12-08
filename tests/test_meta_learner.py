# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.meta_learner import RandomSearchMetaLearner
from mlt.meta_learner import MeanMetaLearner
from mlt.meta_learner import GreedyMetaLearner
from mlt.meta_learner import OptimalMetaLearner
from mlt.meta_learner import run_and_plot_learning_curve
from mlt.data import DAMatrix, NFLDAMatrix, Case2DAMatrix, Case3dDAMatrix

import numpy as np


def test_run_and_plot_learning_curve():
    da_matrix = NFLDAMatrix()
    rs_meta_learner = RandomSearchMetaLearner()
    run_and_plot_learning_curve([rs_meta_learner], da_matrix)


def test_mean_meta_learner():
    n_algos = 13
    thetas = np.arange(n_algos) / n_algos
    # da_matrix = NFLDAMatrix()
    case2_da_matrix = Case2DAMatrix(thetas=thetas)
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    meta_learners = [rs_meta_learner, mean_meta_learner]
    run_and_plot_learning_curve(meta_learners, case2_da_matrix)


def test_all_meta_learners():
    n_algos = 5
    thetas = np.arange(n_algos) / n_algos / 10
    print("thetas:", thetas)
    # da_matrix = NFLDAMatrix()
    case2_da_matrix = Case2DAMatrix(thetas=thetas)
    i_dataset = -2
    print("True perfs:", case2_da_matrix.perfs[i_dataset])
    # print(case2_da_matrix.perfs)
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    greedy_meta_learner = GreedyMetaLearner()
    optimal_meta_learner = OptimalMetaLearner()
    meta_learners = [rs_meta_learner, 
                     mean_meta_learner, 
                     greedy_meta_learner, 
                     optimal_meta_learner]
    run_and_plot_learning_curve(meta_learners, case2_da_matrix, 
                                i_dataset=i_dataset)


def test_case3d_damatrix():
    da_matrix = Case3dDAMatrix()
    i_dataset = -1
    print("True perfs:", da_matrix.perfs[i_dataset])
    # print(case2_da_matrix.perfs)
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    greedy_meta_learner = GreedyMetaLearner()
    optimal_meta_learner = OptimalMetaLearner()
    meta_learners = [rs_meta_learner, 
                     mean_meta_learner, 
                     greedy_meta_learner, 
                     optimal_meta_learner]
    run_and_plot_learning_curve(meta_learners, da_matrix, 
                                i_dataset=i_dataset)


if __name__ == '__main__':
    # test_run_and_plot_learning_curve()
    # test_mean_meta_learner()
    test_all_meta_learners()
    # test_case3d_damatrix()