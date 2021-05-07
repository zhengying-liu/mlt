# Author: Zhengying LIU
# Create: 6 May 2021

from mlt.figures import plot_score_vs_n_tasks_with_error_bars
from mlt.figures import plot_score_vs_n_algos_with_error_bars


def test_plot_score_vs_n_tasks_with_error_bars():
    plot_score_vs_n_tasks_with_error_bars()


def test_plot_score_vs_n_algos_with_error_bars():
    plot_score_vs_n_algos_with_error_bars()


if __name__ == '__main__':
    # test_plot_score_vs_n_tasks_with_error_bars()
    test_plot_score_vs_n_algos_with_error_bars()