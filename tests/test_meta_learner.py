# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.meta_learner import RandomSearchMetaLearner
from mlt.meta_learner import MeanMetaLearner
from mlt.meta_learner import GreedyMetaLearner
from mlt.meta_learner import OptimalMetaLearner
from mlt.meta_learner import run_and_plot_learning_curve, run_leave_one_out
from mlt.meta_learner import run_meta_validation
from mlt.meta_learner import plot_meta_learner_with_different_ranks
from mlt.meta_learner import plot_meta_learner_with_different_true_ranks
from mlt.meta_learner import plot_alc_vs_rank
from mlt.meta_learner import binarize
from mlt.meta_learner import save_fig
from mlt.meta_learner import save_perfs
from mlt.meta_learner import get_the_meta_learners
from mlt.meta_learner import generate_binary_matrix_with_rank
from mlt.meta_learner import run_once_random
from mlt.meta_learner import get_da_matrix_from_real_dataset_dir
from mlt.data import DAMatrix, NFLDAMatrix, Case2DAMatrix, Case3dDAMatrix
from mlt.data import BinarizedMultivariateGaussianDAMatrix

import numpy as np
import matplotlib.pyplot as plt
import os


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

    fig = run_leave_one_out(meta_learners, da_matrix, n_runs=10)

    # Save results
    np.savetxt('perfs.npy', da_matrix.perfs.astype(int), fmt='%i')
    fig.savefig('result.jpg')


def test_run_meta_validation():
    da_matrix = Case3dDAMatrix(n_datasets=2000)
    meta_learners = get_the_meta_learners()
    run_meta_validation(meta_learners, da_matrix)


def run_expe(da_matrix, meta_learners, 
             name_expe=None,
             results_dir='../results', with_once_random=False):
    """Use run_meta_validation to run experiment."""
    if with_once_random:
        fig = run_once_random(da_matrix)
    else:
        fig = None

    fig = run_meta_validation(meta_learners, da_matrix, fig=fig)

    # Create directory for the experiment
    expe_dir = os.path.join(results_dir, str(name_expe))
    os.makedirs(expe_dir, exist_ok=True)

    # Save performance matrix and the figure
    perfs_path = os.path.join(expe_dir, 'perfs.npy')
    fig_path = os.path.join(expe_dir, 'learning-curves.jpg')
    np.savetxt(perfs_path, da_matrix.perfs.astype(int), fmt='%i')
    fig.savefig(fig_path)


def run_nfl():
    n_datasets = 20000
    n_algos = 5
    perfs = (np.random.rand(n_datasets, n_algos) < 0.5).astype(int)
    name_expe = 'nfl'
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3a():
    n_datasets = 20000
    n_algos = 5
    col = (np.random.rand(n_datasets, 1) < 0.5).astype(int)
    perfs = np.concatenate([col] * n_algos, axis=1)
    name_expe = '3a-repeated-columns'
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3b():
    n_datasets = 20000
    n_algos = 2
    X1 = (np.random.rand(n_datasets, 1) < 0.5).astype(int)
    X2 = 1 - X1
    perfs = np.concatenate([X1, X2], axis=1)
    name_expe = '3b-complementary-2-algos'
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3d():
    n_datasets = 20000
    name_expe = '3d'
    da_matrix = Case3dDAMatrix(n_datasets=n_datasets, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3f():
    n_datasets = 20000
    name_expe = '3f'
    epsilon = 1e-1
    A = (np.random.rand(n_datasets * 2, 1) < 0.5 + 2 * epsilon).astype(int)
    B = (np.random.rand(n_datasets * 2, 1) < 0.5 +  epsilon).astype(int)
    C = (np.random.rand(n_datasets * 2, 1) < 0.5 -  epsilon).astype(int)
    D = (np.random.rand(n_datasets * 2, 1) < 0.5 -  2 * epsilon).astype(int)
    perfs = np.concatenate([A, B, C, D], axis=1)

    valid_rows = []
    for row in perfs:
        if not (row[0] == 0 and row[1] == 0 and row[2] == 0):
            valid_rows.append(row)
        if len(valid_rows) == n_datasets:
            break
    perfs = np.array(valid_rows)[:n_datasets]
    assert len(perfs) == n_datasets
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3g():
    n_datasets = 20000
    name_expe = '3g'
    epsilon = 1e-1
    X1 = (np.random.rand(n_datasets, 1) < 0.5 - epsilon).astype(int)
    X2 = 1 - X1
    perfs = np.concatenate([X1, X2], axis=1)
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def test_binarize():
    matrix = np.random.rand(10, 3)
    bm = binarize(matrix)
    print(matrix)
    print(bm)


def test_generate_binary_matrix_with_rank():
    for m in range(1, 10):
        for n in range(1, 10):
            for rank in range(min(m, n) + 1):
                matrix = generate_binary_matrix_with_rank(rank, m, n)
                print(matrix)
                print(rank)
                assert np.linalg.matrix_rank(matrix) == ran


def run_on_real_dataset(dataset_dir):
    """`dataset_dir` should contain a file `*.data` in NumPy format."""
    if os.path.isdir(dataset_dir):
        data_files = [x for x in os.listdir(dataset_dir) 
                        if x.endswith('.data')]
        if len(data_files) != 1:
            raise ValueError("The dataset directory {} ".format(dataset_dir) + 
                                "should contain one `.data` file but got " +
                                "{}.".format(data_files))
        data_file = data_files[0]
        data_path = os.path.join(dataset_dir, data_file)
        name_expe = data_file.split('.')[0]

        # Load real dataset and binarize
        perf = binarize(np.loadtxt(data_path))
        da_matrix = DAMatrix(perfs=perf, name=name_expe)
        meta_learners = get_the_meta_learners(exclude_optimal=True)[1:] # Exclude random

        run_expe(da_matrix, meta_learners=meta_learners, 
                name_expe=name_expe, with_once_random=True)
    else:
        raise ValueError("Not a directory: {}".format(dataset_dir))


def run_on_all_real_datasets(datasets_dir=None):
    if datasets_dir is None:
        datasets_dir = "../datasets"
    
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            run_on_real_dataset(dataset_dir)


def get_real_datasets_dataset_dirs(datasets_dir=None):
    if datasets_dir is None:
        datasets_dir = "../datasets"
    
    dataset_dirs = []
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            dataset_dirs.append(dataset_dir)
    return dataset_dirs


def test_run_once_random():
    da_matrix = Case3dDAMatrix(n_datasets=20000)
    run_once_random(da_matrix, n_meta_learners=10)


def run_leave_one_out_on_real_datasets():
    dataset_dirs = get_real_datasets_dataset_dirs()
    meta_learners = get_the_meta_learners(exclude_optimal=True)[1:]
    for dataset_dir in dataset_dirs:
        name_expe = "LOO-{}".format(os.path.basename(dataset_dir))
        da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
        fig = run_once_random(da_matrix, leave_one_out=True)
        fig = run_leave_one_out(meta_learners, da_matrix, use_all=True, fig=fig)
        save_fig(fig, name_expe=name_expe)
        save_perfs(da_matrix.perfs, name_expe=name_expe)


if __name__ == '__main__':
    # test_run_and_plot_learning_curve()
    # test_mean_meta_learner()
    # test_all_meta_learners()
    # test_case3d_damatrix()
    # test_run_meta_validation()

    # run_3a()
    # run_3b()
    # run_3d()
    # run_3f()
    # run_3g()
    # run_nfl()
    
    # test_binarize()
    # plot_meta_learner_with_different_ranks()
    # test_generate_binary_matrix_with_rank()
    # plot_meta_learner_with_different_true_ranks()
    # plot_alc_vs_rank()

    run_on_all_real_datasets()

    # test_run_once_random()

    # run_leave_one_out_on_real_datasets()