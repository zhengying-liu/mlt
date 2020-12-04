# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.meta_learner import RandomSearchMetaLearner
from mlt.meta_learner import run_and_plot_learning_curve
from mlt.data import DAMatrix, NFLDAMatrix


def test_run_and_plot_learning_curve():
    da_matrix = NFLDAMatrix()
    rs_meta_learner = RandomSearchMetaLearner()
    run_and_plot_learning_curve(rs_meta_learner, da_matrix)


if __name__ == '__main__':
    test_run_and_plot_learning_curve()