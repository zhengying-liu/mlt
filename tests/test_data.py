# Author: Zhengying Liu
# Creation data: 4 Dec 2020

from mlt.data import DAMatrix, NFLDAMatrix
from mlt.data import BinarizedMultivariateGaussianDAMatrix

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
    

if __name__ == '__main__':
    # test_nfldamatrix()
    test_binarized_mg()