# Author: Zhengying LIU
# Create: 6 May 2021

from mlt.data import get_da_matrix_from_real_dataset_dir

from mlt.figures import plot_score_vs_n_tasks_with_error_bars
from mlt.figures import plot_score_vs_n_algos_with_error_bars
from mlt.figures import inspect_da_matrix

import os


def test_plot_score_vs_n_tasks_with_error_bars():
    plot_score_vs_n_tasks_with_error_bars()


def test_plot_score_vs_n_algos_with_error_bars():
    plot_score_vs_n_algos_with_error_bars()


def test_inspect_da_matrix():
    datasets_dir = "../datasets"
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            inspect_da_matrix(da_matrix)


if __name__ == '__main__':
    # test_plot_score_vs_n_tasks_with_error_bars()
    # test_plot_score_vs_n_algos_with_error_bars()
    test_inspect_da_matrix()