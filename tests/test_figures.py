# Author: Zhengying LIU
# Create: 6 May 2021

from mlt import ROOT_DIR

from mlt.data import get_da_matrix_from_real_dataset_dir
from mlt.data import URVDAMatrix

from mlt.figures import plot_score_vs_n_tasks_with_error_bars
from mlt.figures import plot_score_vs_n_algos_with_error_bars
from mlt.figures import inspect_da_matrix
from mlt.figures import plot_all_figures

import os


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


def test_plot_all_figures():
    # dataset_names = ['URV', 'OpenML', 'URV-unnorm']
    # dataset_names = None
    # dataset_names = ['URV-2000']
    dataset_names = ['IndepBetaDist']
    plot_all_figures(dataset_names=dataset_names, log_scale=False, max_ticks=50)


if __name__ == '__main__':
    # test_plot_score_vs_n_tasks_with_error_bars()
    test_plot_score_vs_n_algos_with_error_bars()
    # test_inspect_da_matrix()
    # test_plot_all_figures()