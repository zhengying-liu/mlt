# Author: Zhengying Liu
# Creation data: 4 Dec 2020

from mlt.data import DAMatrix, NFLDAMatrix, ComplementaryDAMatrix
from mlt.data import BetaDistributionDAMatrix
from mlt.data import DirichletDistributionDAMatrix
from mlt.data import BinarizedMultivariateGaussianDAMatrix
from mlt.data import DepUDirichletDistributionDAMatrix
from mlt.data import TransposeDirichletDistributionDAMatrix
from mlt.data import URVDAMatrix
from mlt.data import USVDAMatrix
from mlt.data import TrigonometricPolynomialDAMatrix
from mlt.data import download_autodl_data, parse_autodl_data
from mlt.data import plot_error_bars_empirical_vs_theoretical
from mlt.data import to_df_for_cd_diagram
from mlt.data import get_da_matrix_from_real_dataset_dir
from mlt.data import parse_cepairs_data

from mlt.figures import inspect_da_matrix

import numpy as np

def test_nfldamatrix():
    da_matrix = NFLDAMatrix()
    path_to_dir = da_matrix.save()
    da_matrix2 = DAMatrix.load(path_to_dir)
    print(da_matrix.perfs)
    print(da_matrix2.perfs)

def test_binarized_mg():
    N = 10
    mean = np.arange(N) / N
    C = np.arange(2 * N).reshape(N, 2)
    cov = C.dot(C.T)
    da_matrix = BinarizedMultivariateGaussianDAMatrix(mean, cov, n_datasets=10)
    print(da_matrix.perfs)

def test_complementary():
    da_matrix = ComplementaryDAMatrix()
    print(da_matrix.perfs)


def test_download_autodl_data():
    download_autodl_data()


def test_parse_autodl_data():
    parse_autodl_data(save=True)


def test_plot_error_bars_empirical_vs_theoretical():
    plot_error_bars_empirical_vs_theoretical()


def test_to_df_for_cd_diagram():
    # da_matrix = parse_autodl_data()
    # df = to_df_for_cd_diagram(da_matrix)
    # df.to_csv('autodl.csv', index=False)

    da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir='../datasets/OpenML-Alors')
    df = to_df_for_cd_diagram(da_matrix)
    df.to_csv('openml.csv', index=False)


def test_BetaDistributionDAMatrix():
    n_algos = 20
    delta = np.random.rand(n_algos) * 2    # U[0, 2]
    s = 10                                      # alpha + beta
    alpha = (s + delta) / 2
    alpha = np.sort(alpha)                      # Ascending alpha
    beta = s - alpha
    alpha_beta_pairs = [(alpha[i], beta[i]) for i in range(n_algos)]

    da_matrix = BetaDistributionDAMatrix(alpha_beta_pairs, name='IndepBetaDist')

    da_matrix.save()

    inspect_da_matrix(da_matrix)


def test_NFLBetaDist():
    n_algos = 20
    alpha_beta_pairs = [(5, 5)] * n_algos
    da_matrix = BetaDistributionDAMatrix(alpha_beta_pairs, name='NFLBetaDist')
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_DirichletDistributionDAMatrix():
    alpha = np.arange(20) + 1
    da_matrix = DirichletDistributionDAMatrix(alpha)
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_DepUDirichletDistributionDAMatrix():
    alpha = np.arange(20) + 1
    da_matrix = DepUDirichletDistributionDAMatrix(alpha)
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_TransposeDirichletDistributionDAMatrix():
    n_datasets = 200
    mu = 100
    alpha = mu * (np.arange(n_datasets) + 1)
    da_matrix = TransposeDirichletDistributionDAMatrix(alpha)
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_URVDAMatrix():
    da_matrix = URVDAMatrix(n_datasets=2000, n_algos=2000, normalized=True, name='URV-2000')
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_USVDAMatrix():
    da_matrix = URVDAMatrix(n_datasets=200, n_algos=20, normalized=True, name='USV-0_1-200-20')
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_TrigonometricPolynomialDAMatrix():
    da_matrix = TrigonometricPolynomialDAMatrix(n_datasets=2000, n_algos=20)
    da_matrix.save()
    inspect_da_matrix(da_matrix)


def test_parse_cepairs_data():
    da_matrix = parse_cepairs_data()
    inspect_da_matrix(da_matrix)


if __name__ == '__main__':
    # test_nfldamatrix()
    # test_binarized_mg()
    # test_complementary()
    # test_download_autodl_data()
    # test_parse_autodl_data()
    # test_plot_error_bars_empirical_vs_theoretical()
    # test_to_df_for_cd_diagram()
    # test_BetaDistributionDAMatrix()
    # test_NFLBetaDist()
    # test_DirichletDistributionDAMatrix()
    # test_DepUDirichletDistributionDAMatrix()
    # test_TransposeDirichletDistributionDAMatrix()
    # test_URVDAMatrix()
    # test_USVDAMatrix()
    # test_TrigonometricPolynomialDAMatrix()
    test_parse_cepairs_data()