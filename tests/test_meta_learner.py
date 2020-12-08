# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.meta_learner import RandomSearchMetaLearner
from mlt.meta_learner import MeanMetaLearner
from mlt.meta_learner import GreedyMetaLearner
from mlt.meta_learner import OptimalMetaLearner
from mlt.meta_learner import run_and_plot_learning_curve, run_leave_one_out
from mlt.data import DAMatrix, NFLDAMatrix, Case2DAMatrix, Case3dDAMatrix
from mlt.data import BinarizedMultivariateGaussianDAMatrix

import numpy as np
import matplotlib.pyplot as plt


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


def test_damatrix(ClsDAMatrix, kwargs=None, i_dataset=-1,
                  meta_learners=None, include_optimal=False):
    if kwargs is None:
        kwargs = {}

    da_matrix = ClsDAMatrix(**kwargs)

    print("Performance matrix:", da_matrix.perfs)

    print("True perfs:", da_matrix.perfs[i_dataset])

    if meta_learners is None:
        rs_meta_learner = RandomSearchMetaLearner()
        mean_meta_learner = MeanMetaLearner()
        greedy_meta_learner = GreedyMetaLearner()
        optimal_meta_learner = OptimalMetaLearner()
        meta_learners = [rs_meta_learner, 
                        mean_meta_learner, 
                        greedy_meta_learner,
                        ]
        if include_optimal:
            meta_learners.append(optimal_meta_learner)

    run_and_plot_learning_curve(meta_learners, da_matrix, 
                                i_dataset=i_dataset)


def test_leave_one_out(ClsDAMatrix, kwargs=None,
                  meta_learners=None, include_optimal=False):
    if kwargs is None:
        kwargs = {}

    da_matrix = ClsDAMatrix(**kwargs)

    print("Performance matrix:")
    print(da_matrix.perfs)
    np.savetxt('perfs.npy', da_matrix.perfs.astype(int), fmt='%i')

    if meta_learners is None:
        rs_meta_learner = RandomSearchMetaLearner()
        mean_meta_learner = MeanMetaLearner()
        greedy_meta_learner = GreedyMetaLearner()
        optimal_meta_learner = OptimalMetaLearner()
        meta_learners = [rs_meta_learner, 
                        mean_meta_learner, 
                        greedy_meta_learner,
                        ]
        if include_optimal:
            meta_learners.append(optimal_meta_learner)

    fig = run_leave_one_out(meta_learners, da_matrix)
    fig.savefig('result.jpg')



if __name__ == '__main__':
    # test_run_and_plot_learning_curve()
    # test_mean_meta_learner()
    # test_all_meta_learners()
    # test_case3d_damatrix()

    ClsDAMatrix = BinarizedMultivariateGaussianDAMatrix
    # ClsDAMatrix = Case3dDAMatrix

    # Parameters
    n_algos = 10
    N = n_algos
    # mean = np.arange(N) / N
    mean = np.ones(N) * 0.5
    # rank = 1
    # C = np.arange(rank * N).reshape(N, rank)
    # cov = C.dot(C.T)
    cov = np.eye(N)
    kwargs = {
        'mean': mean,
        'cov': cov,
        'n_datasets': 10000,
        }

    # test_damatrix(BinarizedMultivariateGaussianDAMatrix, kwargs=kwargs,
    #     i_dataset=-1,
    # )
    test_leave_one_out(ClsDAMatrix, kwargs=kwargs)
    # test_leave_one_out(ClsDAMatrix)