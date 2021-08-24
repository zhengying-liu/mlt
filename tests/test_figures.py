# Author: Zhengying LIU
# Create: 6 May 2021

from mlt import ROOT_DIR

from mlt.data import get_da_matrix_from_real_dataset_dir
from mlt.data import URVDAMatrix
from mlt.data import DAMatrix
from mlt.data import SpecialistDAMatrix
from mlt.data import get_all_real_datasets_da_matrix
from mlt.data import TrigonometricPolynomialDAMatrix
from mlt.data import parse_cepairs_data
from mlt.data import BinarizedMultivariateGaussianDAMatrix
from mlt.data import BetaDistributionDAMatrix
from mlt.data import DirichletDistributionDAMatrix
from mlt.data import sample_trigo_polyn

from mlt.figures import plot_score_vs_n_tasks_with_error_bars
from mlt.figures import plot_score_vs_n_algos_with_error_bars
from mlt.figures import inspect_da_matrix
from mlt.figures import plot_all_figures
from mlt.figures import plot_score_vs_n_algos_per_matrix
from mlt.figures import plot_score_vs_n_tasks_per_matrix
from mlt.figures import plot_meta_learner_comparison
from mlt.figures import plot_overfit_curve
from mlt.figures import plot_overfit_curve_sample_test
from mlt.figures import plot_ofc_disjoint_tasks
from mlt.figures import plot_meta_learner_comparison_sample_meta_test
from mlt.figures import plot_full_meta_learner_comparison

from mlt.meta_learner import MeanMetaLearner
from mlt.meta_learner import TopkRankMetaLearner
from mlt.meta_learner import FixedKRankMetaLearner
from mlt.meta_learner import TopPercRankMetaLearner
from mlt.meta_learner import TopKD
from mlt.meta_learner import SRM
from mlt.meta_learner import CountMaxMetaLearner
from mlt.meta_learner import SGDMetaLearner
from mlt.meta_learner import MaxAverageRankMetaLearner

from mlt.metric import ArgmaxMeanMetric
from mlt.metric import AccuracyMetric

from mlt.utils import timer

import os
import numpy as np
import time


DATASETS_DIR = os.path.join(ROOT_DIR, os.pardir, 'datasets')


def test_plot_score_vs_n_tasks_with_error_bars():
    rank = 10
    n_datasets = 2000
    n_algos = 20
    name = "URV-{}-{}-{}".format(rank, n_datasets, n_algos)
    datasets_dir = os.path.join(ROOT_DIR, os.pardir, 'datasets')
    if not os.path.isdir(os.path.join(datasets_dir, name)):
        da_matrix = URVDAMatrix(rank=rank, n_datasets=n_datasets, 
            n_algos=n_algos, name=name)

        # Save the meta-dataset
        da_matrix.save()
    
    dataset_names = [name]
    print("Plotting figures for datasets: {}".format(dataset_names))
    plot_score_vs_n_tasks_with_error_bars(
        datasets_dir=datasets_dir, 
        dataset_names=dataset_names,
        repeat=10,
        max_ticks=20)


def test_plot_score_vs_n_algos_with_error_bars():
    rank = 10
    n_datasets = 200
    n_algos = 20
    name = "URV-{}-{}-{}".format(rank, n_datasets, n_algos)
    datasets_dir = os.path.join(ROOT_DIR, os.pardir, 'datasets')
    if not os.path.isdir(os.path.join(datasets_dir, name)):
        da_matrix = URVDAMatrix(rank=rank, n_datasets=n_datasets, 
            n_algos=n_algos, name=name)

        # Save the meta-dataset
        da_matrix.save()
    
    dataset_names = [name]
    print("Plotting figures for datasets: {}".format(dataset_names))
    plot_score_vs_n_algos_with_error_bars(
        datasets_dir=datasets_dir, 
        dataset_names=dataset_names,
        repeat=10,
        max_ticks=20, shuffling=True, nested=True)


def test_inspect_da_matrix():
    datasets_dir = "../datasets"
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            inspect_da_matrix(da_matrix)


def test_plot_score_vs_n_algos_per_matrix():
    perfs = np.array([
        [1, 0.5, 0],
        [1, 0.5, 0],
        [0, 0.5, 1],
        [0, 0.5, 1],
    ])

    da_matrix = DAMatrix(perfs=perfs, name='Manual')

    meta_learner = MeanMetaLearner()
    plot_score_vs_n_algos_per_matrix(da_matrix, meta_learner, 
        shuffling=False,
        nested=True)


def test_plot_score_vs_n_algos_per_matrix_on_real_datasets():
    meta_learner = MeanMetaLearner()

    datasets_dir="../datasets"
    dataset_names = [
        'URV-10-200-20', 
        'artificial_r50c20r20', 'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    da_matrices = []
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            da_matrices.append(da_matrix)

    for da_matrix in da_matrices:
        plot_score_vs_n_algos_per_matrix(da_matrix, meta_learner, 
            shuffling=True,
            nested=False,
            save=True,
            max_ticks=20)


def test_plot_score_vs_n_tasks_per_matrix_on_real_datasets():
    meta_learner = MeanMetaLearner()

    datasets_dir="../datasets"
    dataset_names = [
        'URV-10-200-20', 
        'artificial_r50c20r20', 'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    da_matrices = []
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            da_matrices.append(da_matrix)

    for da_matrix in da_matrices:
        plot_score_vs_n_tasks_per_matrix(da_matrix, meta_learner, 
            shuffling=True,
            save=True,
            max_ticks=20,
            )


def test_plot_score_vs_n_tasks_per_matrix():
    perfs = np.array([
        [1, 0.5, 0],
        [1, 0.5, 0],
        [0, 0.5, 1],
        [0, 0.5, 1],
    ])

    da_matrix = DAMatrix(perfs=perfs, name='Manual')

    meta_learner = MeanMetaLearner()
    plot_score_vs_n_tasks_per_matrix(da_matrix, meta_learner, shuffling=False)


def test_plot_all_figures():
    # dataset_names = ['URV', 'OpenML', 'URV-unnorm']
    # dataset_names = None
    # dataset_names = ['URV-2000']
    dataset_names = ['IndepBetaDist']
    plot_all_figures(dataset_names=dataset_names, log_scale=False, max_ticks=50)


def test_plot_meta_learner_comparison():
    top10_ml = FixedKRankMetaLearner(k=10)
    top1_ml = FixedKRankMetaLearner(k=1)
    infty_ml = FixedKRankMetaLearner(k=10**5)
    top10perc_ml = TopPercRankMetaLearner(perc=10)
    cv_ml = TopkRankMetaLearner()
    top_k_d = TopKD()
    srm = SRM()
    meta_learners = [
        top1_ml, 
        cv_ml, 
        top10_ml,  
        top10perc_ml, 
        infty_ml, 
        top_k_d, 
        srm,
        ]

    datasets_dir="../datasets"
    dataset_names = ['artificial_r50c20r20', 'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    da_matrices = []
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            da_matrices.append(da_matrix)

    perfs = np.arange(100).reshape(10, 10)
    da_matrix = DAMatrix(perfs=perfs, name="Always-9")
    da_matrices.append(da_matrix)

    # Generalist
    n_algos = 20
    alpha1 = np.arange(n_algos) + 1
    da_matrix = SpecialistDAMatrix(alphas=[alpha1], name='Generalist')
    da_matrices.append(da_matrix)

    # Specialist
    alpha2 = n_algos - np.arange(n_algos) + 1
    da_matrix = SpecialistDAMatrix(alphas=[alpha1, alpha2], name='Specialist2')
    da_matrices.append(da_matrix)

    for da_matrix in da_matrices:
        n_datasets = len(da_matrix.datasets)
        print("Meta-dataset:", da_matrix.name)
        print("n_datasets:", n_datasets)
        da_tr, da_te = da_matrix.train_test_split()
        plot_meta_learner_comparison(da_tr, da_te, meta_learners, repeat=100)


def test_plot_overfit_curve():
    datasets_dir="../datasets"
    dataset_names = ['artificial_r50c20r20', 'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    da_matrices = []
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            da_matrices.append(da_matrix)
    for da_matrix in da_matrices:
        n_datasets = len(da_matrix.datasets)
        print("Meta-dataset:", da_matrix.name)
        print("n_datasets:", n_datasets)
        da_tr, da_te = da_matrix.train_test_split()
        plot_overfit_curve(da_tr, da_te)


def test_SRM():
    da_matrices = get_all_real_datasets_da_matrix()
    for da_matrix in da_matrices:
        n_datasets = len(da_matrix.datasets)
        print("Meta-dataset:", da_matrix.name)
        print("n_datasets:", n_datasets)
        da_tr, da_te = da_matrix.train_test_split()
        plot_overfit_curve(da_tr, da_te)


def test_plot_overfit_curve_sample_test():
    da_matrices = get_all_real_datasets_da_matrix()

    # Generalist
    da_matrix = URVDAMatrix(n_datasets=10000, n_algos=20, name='URV10000')
    da_matrices.append(da_matrix)

    for da_matrix in da_matrices:
        n_datasets = len(da_matrix.datasets)
        print("Meta-dataset:", da_matrix.name)
        print("n_datasets:", n_datasets)
        plot_overfit_curve_sample_test(da_matrix)


def test_plot_ofc_disjoint_tasks():
    da_matrices = get_all_real_datasets_da_matrix()

    # Generalist
    da_matrix = URVDAMatrix(n_datasets=10000, n_algos=20, name='URV10000')
    da_matrices.append(da_matrix)

    for da_matrix in da_matrices:
        n_datasets = len(da_matrix.datasets)
        print("Meta-dataset:", da_matrix.name)
        print("n_datasets:", n_datasets)
        plot_ofc_disjoint_tasks(da_matrix, n_tasks_per_split=5)


def test_plot_meta_learner_comparison_sample_meta_test():
    # Meta-learners
    ml_mean = MeanMetaLearner(name='mean')
    ml_srm = SRM()
    ml_cm = CountMaxMetaLearner()
    ml_topk = TopkRankMetaLearner()
    ml_sgd = SGDMetaLearner()
    ml_mar = MaxAverageRankMetaLearner()
    meta_learners = [
        ml_mean,
        ml_srm,
        ml_cm,
        ml_topk,
        ml_sgd,
        ml_mar,
    ]
    # Metric
    metric = ArgmaxMeanMetric()
    # Configurations
    n_datasets = 100
    n_algos = 20
    repeat = 100
    train_size = 0.5
    # Use TrigoPolyn
    da_matrices = []
    for i in range(5):
        da_matrix = TrigonometricPolynomialDAMatrix(
            n_datasets=n_datasets, 
            n_algos=n_algos,
            name='TrigoPolyn-{}'.format(i))
        da_matrices.append(da_matrix)

        
    # Real datasets 
    real_datasets = get_all_real_datasets_da_matrix()
    da_matrices += real_datasets

    # CEpairs 
    da_cepairs = parse_cepairs_data()
    da_matrices.append(da_cepairs)

    for da_matrix in da_matrices:
        plot_meta_learner_comparison_sample_meta_test(
            da_matrix, 
            meta_learners, 
            metric=metric,
            repeat=repeat,
            train_size=train_size,
            save=True,
            show=False,
        )


@timer
def test_plot_full_meta_learner_comparison():
    # Meta-learners
    ml_mean = MeanMetaLearner(name='mean')
    ml_cm = CountMaxMetaLearner(name='count-max')
    ml_mar = MaxAverageRankMetaLearner('max-avg-rank')
    meta_learners = [
        ml_mean,
        ml_cm,
        ml_mar,
    ]

    inspect = True

    # Real-world datasets
    real_das = get_all_real_datasets_da_matrix()
    # CEpairs 
    da_cepairs = parse_cepairs_data()
    real_das.append(da_cepairs)
    del real_das[4] # Remove artificial_r50c20r20
    # del real_das[4] # Remove CEpairs

    synt_das = []
    # Independent Gaussian
    n_datasets = 76
    n_algos = 292
    mean = [j / n_algos for j in range(1, n_algos + 1)]
    cov = np.eye(n_algos)
    indep_gauss = BinarizedMultivariateGaussianDAMatrix(
        mean, cov, 
        n_datasets=n_datasets,
        binarized=False, 
        name='IndepGauss'
    )
    indep_gauss.set_best_algo(n_algos - 1)
    synt_das.append(indep_gauss)

    # Multi-variate Guassian
    n_datasets = 15
    n_datasets = 100
    n_algos = 20
    mean = [j / n_algos for j in range(1, n_algos + 1)]
    M = np.eye(n_algos)
    for j in range(n_algos):
        for i in range(j + 1):
            M[i, j] = 1
    cov = M.dot(M.T) / (n_algos)
    multi_gauss = BinarizedMultivariateGaussianDAMatrix(
        mean, cov, 
        n_datasets=n_datasets, 
        binarized=False, 
        name='MultiGauss'
    )
    multi_gauss.set_best_algo(n_algos - 1)
    synt_das.append(multi_gauss)

    # Beta distribution
    n_datasets = 22
    n_algos = 24
    alpha_beta_pairs = [(j, n_algos + 1 - j) for j in range(1, n_algos + 1)]
    indep_beta = BetaDistributionDAMatrix(
        alpha_beta_pairs,
        n_datasets=n_datasets,
        name='IndepBeta',
    )
    indep_beta.set_best_algo(n_algos - 1)
    synt_das.append(indep_beta)

    # Dirichlet distribution
    n_datasets = 30
    n_algos = 17
    alpha = [j for j in range(1, n_algos + 1)]
    dirich = DirichletDistributionDAMatrix(
        alpha,
        n_datasets=n_datasets,
        name='Dirichlet',
    )
    dirich.set_best_algo(n_algos - 1)
    synt_das.append(dirich)

    # TrigoPolyn
    n_datasets = 8608
    n_algos = 163
    funcs = sample_trigo_polyn(A=n_algos, K=5)
    for j, f in enumerate(funcs):
        f.coeffs[0] = (j + 1) / n_algos
    trigo_polyn = TrigonometricPolynomialDAMatrix(
        funcs=funcs,
        n_datasets=n_datasets,
    )
    trigo_polyn.set_best_algo(n_algos - 1)
    synt_das.append(trigo_polyn)

    if inspect:
        for da in real_das:
            inspect_da_matrix(da)
        for da in synt_das:
            inspect_da_matrix(da)

    # Configuration
    repeat = 100
    train_size = 0.5
    
    # plot_full_meta_learner_comparison(real_das, meta_learners,
    #     repeat=repeat,
    #     train_size=train_size,
    #     save=True,
    #     show=True,
    # )
    name_metric = 'accuracy (Acc)'

    # plot_full_meta_learner_comparison(synt_das, meta_learners,
    #     repeat=repeat,
    #     train_size=train_size,
    #     save=True,
    #     show=True,
    #     metric=AccuracyMetric(name=name_metric),
    #     ylabel=name_metric,
    # )



if __name__ == '__main__':
    # test_plot_score_vs_n_tasks_with_error_bars()
    # test_plot_score_vs_n_algos_with_error_bars()
    # test_inspect_da_matrix()
    # test_plot_all_figures()
    # test_plot_score_vs_n_algos_per_matrix()
    # test_plot_score_vs_n_tasks_per_matrix()
    # test_plot_meta_learner_comparison()
    # test_plot_score_vs_n_algos_per_matrix_on_real_datasets()
    # test_plot_score_vs_n_tasks_per_matrix_on_real_datasets()
    # test_plot_overfit_curve()
    # test_plot_overfit_curve_sample_test()
    # test_plot_ofc_disjoint_tasks()
    # test_plot_meta_learner_comparison_sample_meta_test()
    test_plot_full_meta_learner_comparison()