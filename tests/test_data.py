# Author: Zhengying Liu
# Creation data: 4 Dec 2020

from mlt.data import DAMatrix, NFLDAMatrix

def test_nfldamatrix():
    da_matrix = NFLDAMatrix()
    path_to_dir = da_matrix.save()
    da_matrix2 = DAMatrix.load(path_to_dir)
    print(da_matrix.perfs)
    print(da_matrix2.perfs)
    

if __name__ == '__main__':
    test_nfldamatrix()