# Author: Zhengying LIU
# Create: 6 May 2021

from mlt import ROOT_DIR

from mlt.data import get_da_matrix_from_real_dataset_dir
from mlt.data import URVDAMatrix
from mlt.data import DAMatrix

from mlt.figures import plot_score_vs_n_tasks_with_error_bars
from mlt.figures import plot_score_vs_n_algos_with_error_bars
from mlt.figures import inspect_da_matrix
from mlt.figures import plot_all_figures
from mlt.figures import plot_score_vs_n_algos_per_matrix
from mlt.figures import plot_score_vs_n_tasks_per_matrix
from mlt.figures import plot_meta_learner_comparison

from mlt.meta_learner import MeanMetaLearner
from mlt.meta_learner import TopkRankMetaLearner
from mlt.meta_learner import FixedKRankMetaLearner
from mlt.meta_learner import TopPercRankMetaLearner

import os
import numpy as np


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
    top10perc_ml = TopPercRankMetaLearner(perc=10)
    cv_ml = TopkRankMetaLearner()
    meta_learners = [cv_ml, top10_ml, top10perc_ml]

    datasets_dir="../datasets"
    dataset_names = ['artificial_r50c20r20', 'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            n_datasets = len(da_matrix.datasets)
            print("Meta-dataset:", da_matrix.name)
            print("n_datasets:", n_datasets)
            da_tr, da_te = da_matrix.train_test_split()
            plot_meta_learner_comparison(da_tr, da_te, meta_learners, repeat=100)
            


if __name__ == '__main__':
    # test_plot_score_vs_n_tasks_with_error_bars()
    # test_plot_score_vs_n_algos_with_error_bars()
    # test_inspect_da_matrix()
    # test_plot_all_figures()
    # test_plot_score_vs_n_algos_per_matrix()
    # test_plot_score_vs_n_tasks_per_matrix()
    test_plot_meta_learner_comparison()