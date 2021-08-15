# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt import ROOT_DIR
from mlt.utils import download_file_from_google_drive
from scipy.stats import ortho_group

import ast
import json
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd

np.random.seed(40)


class DAMatrix(object):
    """A DAMatrix object contains a performance matrix, given a list of datasets
    and a list of algorithms. Each row corresponds to a dataset and each column
    corresponds to an algorithm. Each entry (which could be `None`) of i-th row
    and j-th column corresponds to the performance of executing j-th algorithm
    on i-th dataset.
    """

    def __init__(self, perfs=None, datasets=None, algos=None, name=None, 
            negative_score=False):
        """
        Args:
          perfs: a 2-D NumPy array of shape (`n_datasets`, `n_algos`)
          datasets: a list of BetaDataset objects
          algos: a list of BetaAlgo objects
          name: str, name of the DAMatrix
          negative_score: boolean. If True, the smaller performance score is,
            the better.
        """
        self.datasets = datasets if datasets else []
        self.algos = algos if algos else []
        self.name = str(name)
        self.negative_score = negative_score

        # Initialize empty performance matrix if not provided
        if perfs is None:
            n_datasets = len(self.datasets)
            n_algos = len(self.algos)
            shape = (n_datasets, n_algos)
            self.perfs = np.full(shape, np.nan)
        else:
            self.perfs = perfs
            n_datasets, n_algos = perfs.shape
            datasets, algos = get_anonymized_lists(n_datasets, n_algos)
            if not self.datasets:
                self.datasets = datasets
            if not self.algos:
                self.algos = algos

    def append_dataset(self, dataset):
        pass

    def append_algo(self, algo):
        pass

    def eval(self, i_dataset, i_algo):
        entry = self.perfs[i_dataset][i_algo]
        if entry is None or entry is np.nan:
            algo = self.algos[i_algo]
            dataset = self.datasets[i_dataset]
            self.perfs[i_dataset][i_algo] = algo.run(dataset)
            return self.perfs[i_dataset][i_algo]
        else:
            return entry

    def save(self, path_to_dir=None):
        """Serialize the DA matrix to `path_to_dir`.
        
        Returns:
          path_to_dir: str, the path to the directory containing the DA matrix.
        """
        # Default directory containing generated datasets is 
        #   '<mlt REPO>/datasets/'
        if path_to_dir is None:
            try:
                pwd = os.path.dirname(__file__)
            except:
                pwd = '.'
            path_to_datasets = os.path.join(pwd, os.pardir, 'datasets')
            path_to_dir = os.path.join(path_to_datasets, self.name)
            path_to_dir = os.path.abspath(path_to_dir)
            if not os.path.isdir(path_to_dir):
                os.makedirs(path_to_dir, exist_ok=True)
        
        # Save performance matrix
        perfs_filename = "{}.data".format(self.name)
        perfs_path = os.path.join(path_to_dir, perfs_filename)
        np.savetxt(perfs_path, self.perfs)

        # Save list of datasets
        datasets_filename = "{}.datasets".format(self.name)
        datasets_path = os.path.join(path_to_dir, datasets_filename)
        with open(datasets_path, 'w') as f:
            f.write(str([d.name for d in self.datasets]))

        # Save list of algos
        algos_filename = "{}.algos".format(self.name)
        algos_path = os.path.join(path_to_dir, algos_filename)
        with open(algos_path, 'w') as f:
            f.write(str([a.name for a in self.algos]))

        print("Successfully saved DAMatrix {} to {}."\
            .format(self.name, path_to_dir))

        return path_to_dir

    @classmethod
    def load(cls, path_to_dir):
        # Load DAMatrix
        files = os.listdir(path_to_dir)
        ext_name = '.data'
        perfs_files = [x for x in files if x.endswith(ext_name)]
        if len(perfs_files) == 0:
            raise FileNotFoundError("No `{}` file no in {}"\
                .format(ext_name, path_to_dir))
        elif len(perfs_files) > 1:
            raise FileNotFoundError("Multiple `{}` files found in {}: {}"\
                .format(ext_name, path_to_dir, perfs_files))
        
        perfs_file = perfs_files[0]
        name = perfs_file[:-len(ext_name)]
        perfs_path = os.path.join(path_to_dir, perfs_file)
        perfs = np.loadtxt(perfs_path)

        # load datasets and algos
        datasets_file = "{}.datasets".format(name)
        datasets_path = os.path.join(path_to_dir, datasets_file)
        if os.path.isfile(datasets_path):
            with open(datasets_path, 'r') as f:
                line = f.readline()
                datasets = ast.literal_eval(line)
        else:
            datasets = None
        algos_file = "{}.algos".format(name)
        algos_path = os.path.join(path_to_dir, algos_file)
        if os.path.isfile(algos_path):
            with open(algos_path, 'r') as f:
                line = f.readline()
                algos = ast.literal_eval(line)
        else:
            algos = None

        return DAMatrix(perfs=perfs, datasets=datasets, algos=algos, name=name)


    def train_test_split(self, train_size=0.5, shuffling=True):
        """Split the matrix """
        n_datasets = len(self.datasets)

        if isinstance(train_size, float):
            if not (train_size <= 1 and train_size > 0):
                raise ValueError("`train_size` should be in (0, 1].")
            train_size = int(n_datasets * train_size)

        if shuffling:
            indices_train = np.random.choice(n_datasets, train_size, 
                replace=False)
            indices_test = [i for i in range(n_datasets) if not i in set(indices_train)]
        else:
            indices_train = range(train_size)
            indices_test = range(train_size, n_datasets)
        

        perfs = self.perfs[indices_train, :]
        datasets = list(np.array(self.datasets)[indices_train])
        algos = self.algos
        name = self.name + "-meta-train"
        da_meta_train = DAMatrix(perfs=perfs, datasets=datasets, 
                                algos=algos, name=name)

        perfs = self.perfs[indices_test, :]
        datasets = list(np.array(self.datasets)[indices_test])
        algos = self.algos
        name = self.name + "-meta-test"
        da_meta_test = DAMatrix(perfs=perfs, datasets=datasets, 
                                algos=algos, name=name)
        # Transfer info on the ground truth / best algorithm
        if hasattr(self, 'best_algo'):
            da_meta_train.best_algo = self.best_algo
            da_meta_test.best_algo = self.best_algo
        
        return da_meta_train, da_meta_test


    def get_algo_subset(self, indices_algo):
        perfs = self.perfs[:, indices_algo]
        datasets = self.datasets
        algos = list(np.array(self.algos)[indices_algo])
        name = self.name + "-algo-subset"
        da_algo_subset = DAMatrix(perfs=perfs, datasets=datasets, 
                                algos=algos, name=name)
        return da_algo_subset
        

class NFLDAMatrix(DAMatrix):

    def __init__(self, n_datasets=1000, n_algos=13, theta=0.5, 
                 name="NFLDAMatrix"):
        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        self.theta = theta

        DAMatrix.__init__(self, datasets=datasets, algos=algos, name=name)

        for i in range(n_datasets):
            for j in range(n_algos):
                self.eval(i, j)

    def eval(self, i_dataset, i_algo):
        entry = self.perfs[i_dataset][i_algo]
        if np.isnan(entry) or entry is None:
            # Bernoulli distribution with parameter `self.theta`
            entry = int(np.random.rand() < self.theta)
            self.perfs[i_dataset][i_algo] = entry
            return entry
        else:
            return entry


class Case2DAMatrix(DAMatrix):

    def __init__(self, n_datasets=1000, thetas=None, name='Case2DAMatrix'):
        """
        Args:
          n_datasets: number datasets in the DA matrix
          thetas: list of float, Bernoulli parameters for each algorithm
          name: str, name of the DA matrix
        """
        n_algos = len(thetas)

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        self.thetas = thetas

        DAMatrix.__init__(self, datasets=datasets, algos=algos, name=name)
        
        for i in range(n_datasets):
            for j in range(n_algos):
                self.eval(i, j)

    def eval(self, i_dataset, i_algo):
        entry = self.perfs[i_dataset][i_algo]
        if np.isnan(entry) or entry is None:
            # Bernoulli distribution with parameter `self.theta`
            entry = int(np.random.rand() < self.thetas[i_algo])
            self.perfs[i_dataset][i_algo] = entry
            return entry
        else:
            return entry


class Case3dDAMatrix(DAMatrix):

    def __init__(self, n_datasets=10000, name='Case3d_DAMatrix'):
        """Example 3.d. in the MLT paper.
        """
        epsilon = 1e-1
        self.thetas = [0.5 + epsilon, 0.5 - epsilon, 0.5]
        n_algos = 4

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        DAMatrix.__init__(self, datasets=datasets, algos=algos, name=name)
        
        for i in range(n_datasets):
            for j in range(n_algos):
                self.eval(i, j)

    def eval(self, i_dataset, i_algo):
        entry = self.perfs[i_dataset][i_algo]
        if np.isnan(entry) or entry is None:
            # Bernoulli distribution with parameter `self.theta`
            if i_algo == 3:
                entry = 1 - self.perfs[i_dataset][2]
            else:
                entry = int(np.random.rand() < self.thetas[i_algo])
            self.perfs[i_dataset][i_algo] = entry
            return entry
        else:
            return entry


class ComplementaryDAMatrix(DAMatrix):

    def __init__(self, cardinal_clique=2,
                 n_datasets=20000, 
                 n_algos=5,
                 name='ComplementaryDAMatrix',
                 shuffle_column=False,
                 ):
        if cardinal_clique > n_algos:
            raise ValueError("The clique cardinal {} ".format(cardinal_clique) +
                             "should be less than n_algos={}.".format(n_algos))
        
        clique_indices = np.random.randint(cardinal_clique, size=n_datasets)
        
        clique_cols = np.zeros(shape=(n_datasets, cardinal_clique)).astype(int)
        for i, idx in enumerate(clique_indices):
            clique_cols[i][idx] = 1

        if n_algos > cardinal_clique:
            other_cols = np.random.rand(n_datasets, n_algos - cardinal_clique)
            other_cols = (other_cols < 0.5).astype(int)
            all_cols = np.concatenate([clique_cols, other_cols], axis=1)
        else:
            all_cols = clique_cols

        if shuffle_column:
            permutation = np.random.permutation(n_algos)
            idx = np.empty_like(permutation)
            idx[permutation] = np.arange(len(permutation))
            all_cols[:] = all_cols[:, idx]

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)
        
        DAMatrix.__init__(self, 
            perfs=all_cols,
            datasets=datasets, 
            algos=algos, 
            name=name, 
            )


class CopulaCliqueDAMatrix(DAMatrix):
    """Clique: a list one column is equal to 1. All marginal dist are B(0.5).
    """

    def __init__(self, cardinal_clique=2,
                 n_datasets=20000, 
                 n_algos=5,
                 name='CopulaCliqueDAMatrix',
                 shuffle_column=False,
                 ):
        if cardinal_clique > n_algos:
            raise ValueError("The clique cardinal {} ".format(cardinal_clique) +
                             "should be less than n_algos={}.".format(n_algos))
        
        clique_indices = np.random.randint(cardinal_clique, size=n_datasets)
        
        clique_cols = np.zeros(shape=(n_datasets, cardinal_clique)).astype(int)
        for i, idx in enumerate(clique_indices):
            clique_cols[i][idx] = 1
            for j in range(cardinal_clique):
                if j != idx:
                    target_theta = 0.5
                    # proba to make the marginal uniform, i.e. B(0.5)
                    proba = (target_theta * cardinal_clique - 1) / (cardinal_clique - 1)
                    clique_cols[i][j] = int(np.random.rand() < proba)

        if n_algos > cardinal_clique:
            other_cols = np.random.rand(n_datasets, n_algos - cardinal_clique)
            other_cols = (other_cols < 0.5).astype(int)
            all_cols = np.concatenate([clique_cols, other_cols], axis=1)
        else:
            all_cols = clique_cols

        if shuffle_column:
            permutation = np.random.permutation(n_algos)
            idx = np.empty_like(permutation)
            idx[permutation] = np.arange(len(permutation))
            all_cols[:] = all_cols[:, idx]

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)
        
        DAMatrix.__init__(self, 
            perfs=all_cols,
            datasets=datasets, 
            algos=algos, 
            name=name, 
            )



class BinarizedMultivariateGaussianDAMatrix(DAMatrix):

    def __init__(self, mean, cov, n_datasets=1000, 
                 name='BinarizedMultivariateGaussian'):
        """
        Args:
          mean: 1-D array, mean vector of the Multi-vriate Gaussian 
            random variable (MGRV)
          cov: 2-D array, covariance matrix of the MGRV
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
        """
        n_algos = len(mean)

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        perfs = np.random.multivariate_normal(mean, cov, size=n_datasets)
        threshold = np.median(perfs)
        binarized_perfs = (perfs > threshold).astype(int)

        DAMatrix.__init__(self, 
            perfs=binarized_perfs,
            datasets=datasets, 
            algos=algos, 
            name=name, 
            )


class BetaDistributionDAMatrix(DAMatrix):

    def __init__(self, alpha_beta_pairs, n_datasets=2000,
                 name='BetaDist'):
        """
        Args:
          alpha_beta_pairs: list of tuples of the form (alpha_i, beta_i), the
              parameters of the beta distribution of each column
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
          name: str, name of the DA matrix
        """
        n_algos = len(alpha_beta_pairs)

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        cols = []

        for alpha, beta in alpha_beta_pairs:
            col = np.random.beta(alpha, beta, size=n_datasets)
            cols.append(col)

        perfs = np.array(cols).T

        DAMatrix.__init__(self, 
            perfs=perfs,
            datasets=datasets, 
            algos=algos,
            name=name, 
            )


class DirichletDistributionDAMatrix(DAMatrix):

    def __init__(self, alpha, n_datasets=2000, name="DirichletDist"):
        """
        Args:
          alpha: list of floats, the parameters of the Dirichlet distribution.
            Has `n_algos` as length.
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
          name: str, name of the DA matrix
        """
        n_algos = len(alpha)

        perfs = np.random.dirichlet(alpha, size=n_datasets)

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)


class SpecialistDAMatrix(DAMatrix):
    """Several groups (domains, modalities, etc) of tasks exist"""

    def __init__(self, alphas, n_datasets=2000, name="Specialist", shuffling=True):
        """
        Args:
          alphas: lists of floats, the parameters of the Dirichlet distribution.
            Has `n_algos` as length. Each `alpha` should have the same length.
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
          name: str, name of the DA matrix
        """
        if len(alphas) == 0:
            raise ValueError("`alphas` should contain at least one list as element.")
        n_groups = len(alphas)
        n_algos = len(alphas[0])

        perfss = []

        for i, alpha in enumerate(alphas):
            if i == 0:
                size = n_datasets - (n_datasets // n_groups) * (n_groups - 1)
            else:
                size = n_datasets // n_groups
            perfs = np.random.dirichlet(alpha, size=size)
            perfss.append(perfs)
        
        perfs = np.concatenate(perfss, axis=0)

        if shuffling:
            perm = np.random.permutation(n_datasets)
            perfs = perfs[perm, :]

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)


class TransposeDirichletDistributionDAMatrix(DAMatrix):

    def __init__(self, alpha, n_algos=20, name="TransDirichletDist"):
        """
        Args:
          alpha: list of floats, the parameters of the Dirichlet distribution.
            Has `n_datasets` aas length.
          n_algos: int, number of algorithms (i.e. columns) in the DA matrix
          name: str, name of the DA matrix
        """
        n_datasets = len(alpha)

        perfs = np.random.dirichlet(alpha, size=n_algos).T

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)


class URVDAMatrix(DAMatrix):

    def __init__(self, rank=20, U=None, V=None, n_datasets=100, n_algos=20, 
                 name="URV", normalized=True):
        """Performance matrix of the form URV where R is a random matrix of 
        standard Gaussian entries. If `normalized`, a standard logistic function
        (1 / (1 + e^(-x)) is applied to all entries in the end.

        Args:
          rank: R will be a square matrix of shape (rank, rank).
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
          n_algos: int, number of algorithms (i.e. columns) in the DA matrix
          U: NumPy array of shape (n_datasets, rank)
          V: NumPy array of shape (rank, n_algos)
          name: str, name of the DA matrix
          normalized: 
        """
        R = np.random.randn(rank, rank)

        if U is None:
            U = np.random.randn(n_datasets, rank)
        
        if V is None:
            V = np.random.randn(rank, n_algos)

        perfs = U.dot(R).dot(V)

        if normalized:
            perfs = 1 / (1 + np.exp(-perfs))

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)


class USVDAMatrix(DAMatrix):

    def __init__(self, tau=0.1, n_datasets=200, n_algos=20, U=None, V=None,
                 name="USV", normalized=True):
        """Performance matrix of the form USV where S is a diagonal matrix. The
        i-th entry of S is exp(- i * tau).
        U and V should be orthogonal matrix.
        If `normalized`, a standard logistic function
        (1 / (1 + e^(-x)) is applied to all entries in the end.

        Args:
          tau: float, the i-th entry of S is exp(- i * tau)
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
          n_algos: int, number of algorithms (i.e. columns) in the DA matrix
          U: NumPy array of shape (n_datasets, rank)
          V: NumPy array of shape (rank, n_algos)
          name: str, name of the DA matrix
          normalized: 
        """
        rank = min(n_datasets, n_algos)

        if U is None:
            U = ortho_group.rvs(n_datasets)[:, :rank]
        
        if V is None:
            V = ortho_group.rvs(n_algos)[:rank, :]

        v = np.exp(-tau * np.arange(rank))
        S = np.diag(v)

        perfs = U.dot(S).dot(V)

        if normalized:
            perfs = 1 / (1 + np.exp(-perfs))

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)


###################################################
##### Trigonometric Polynomials Meta-datasets #####
###################################################

class TrigonometricPolynomial(object):
    
    def __init__(self, coeffs):
        """Real-valued function defined by trigonometric sums with `coeffs` as coefficents. If
            coeffs = [a0, a1, b1, a2, b2, ...],
        then the defined function is given by
            a0 + a1*cos(2*pi*x) + b1*sin(2*pi*x) + a2*cos(2*2*pi*x) + b2*sin(2*2*pi*x) + ...
        or
            a0 + sum_k [a_k*cos(2*k*pi*x) + b_k*sin(2*k*pi*x)].
        
        Args:
          coeffs: list of float, the coefficients of the trigonometric polynomial.
        """
        self.coeffs = coeffs
        
    def __call__(self, x):
        if len(self.coeffs) == 0:
            return 0
        res = self.coeffs[0]    # a0
        le = len(self.coeffs)
        for k in range(1, le//2 + 1):
            ak = self.coeffs[2 * k - 1]
            bk = self.coeffs[2 * k] if 2 * k < le else 0
            res += ak * np.cos(2 * k * np.pi * x) + bk * np.sin(2 * k * np.pi * x)
        return res


def sample_trigo_polyn(A=20, K=5):
    funcs = [None] * A
    for a in range(A):
        coeffs = np.random.random(2 * K + 1)
        scale = 1 / (np.arange(2 * K + 1) + 1)
        coeffs *= scale
        func = TrigonometricPolynomial(coeffs)
        funcs[a] = func
    return funcs


class TrigonometricPolynomialDAMatrix(DAMatrix):

    def __init__(self, funcs=None, n_datasets=2000, n_algos=20, 
            name="TrigoPolyn"):
        if funcs is None:
            funcs = sample_trigo_polyn(A=n_algos)
        else:
            n_algos = len(funcs)
        self.funcs = funcs
        perfs = np.zeros((n_datasets, n_algos))
        for i_d in range(n_datasets):
            x = np.random.rand()
            for i_a in range(n_algos):
                perfs[i_d, i_a] = funcs[i_a](x)
        
        expected_values = [f.coeffs[0] for f in funcs]
        self.best_algo = np.argmax(expected_values)

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)
        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)

    def get_funcs(self):
        return self.funcs


class DepUDirichletDistributionDAMatrix(DAMatrix):

    def __init__(self, alpha, n_datasets=2000, name="DepUDirichletDist"):
        """Scale every column to have 0.5 as average. 
        
        TODO: Maximum thresholded to 1.

        Args:
          alpha: list of floats, the parameters of the Dirichlet distribution
          n_datasets: int, number of datasets (i.e. rows) in the DA matrix
          name: str, name of the DA matrix
        """
        n_algos = len(alpha)

        perfs = np.random.dirichlet(alpha, size=n_datasets)

        for i in range(n_algos):
            perfs[:, i] = perfs[:, i] / np.mean(perfs[:, i]) / 2

        datasets, algos = get_anonymized_lists(n_datasets, n_algos)

        super().__init__(perfs=perfs, datasets=datasets, algos=algos, name=name)


class BetaAlgo(object):

    def __init__(self, name=None, code=None):
        self.name = name
        self.code = code

    def run(self, dataset):
        pass

    def __repr__(self):
        return "BetaAlgo(name={}, code={})".format(self.name, self.code)

    def __str__(self):
        return str(self.name)


class BetaDataset(object):

    def __init__(self, name=None, data=None, metadata=None):
        self.name = name
        self.data = data
        self.metadata = metadata

    def __repr__(self):
        s = "BetaDataset(name={}, data={}, metadata={})"\
            .format(self.name, self.data, self.metadata)
        return s


def get_anonymized_lists(n_datasets, n_algos):
    """Given number of datasets `n_datasets` and number of algorithms 
    `n_algos`, return one list of anonymized BetaDataset objects and 
    one list of anonymizd BetaAlgo objects.

    Args:
      n_datasets: int, number of datasets
      n_algos: int, number of algorithms

    Returns:
      datasets: list of BetaDataset objects, each dataset having a name 
        'Dataset i'
      algos: list of BetaAlgo objects, each algorithm having a name 
        'Algorithm i'
    """
    datasets = []
    for i in range(n_datasets):
        dataset_name = "Dataset {}".format(i)
        dataset = BetaDataset(name=dataset_name)
        datasets.append(dataset)

    algos = []
    for i in range(n_algos):
        algo_name = "Algorithm {}".format(i)
        algo = BetaAlgo(name=algo_name)
        algos.append(algo)

    return datasets, algos


def get_da_matrix_from_real_dataset_dir(dataset_dir):
    if os.path.isdir(dataset_dir):
        da_matrix = DAMatrix.load(dataset_dir)
        return da_matrix
    else:
        raise ValueError("Not a directory: {}".format(dataset_dir))


def get_all_real_datasets_da_matrix(datasets_dir=None):
    """Get DA matrices for the datasets:
        - artificial_r50c20r20
        - AutoDL
        - AutoML
        - OpenML-Alors
        - Statlog
    You need to guarantee that these datasets are indeed contained in
    `datasets_dir`.
    """
    if datasets_dir is None:
        datasets_dir = os.path.join(ROOT_DIR, os.pardir, 'datasets')
        print("No datasets_dir given. Using datasets_dir={}"\
            .format(datasets_dir))
    dataset_names = ['artificial_r50c20r20', 'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    da_matrices = []
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            da_matrices.append(da_matrix)
    return da_matrices


def download_autodl_data(filename='all_results.csv', save_dir=None):
    if save_dir is None:
        try:
            pwd = os.path.dirname(__file__)
        except:
            pwd = '.'
        save_dir = os.path.join(pwd, os.pardir, 'datasets', 'AutoDL')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.abspath(os.path.join(save_dir, filename))
    if os.path.isfile(filepath):
        print("Data already downloaded at:\n\t{}\nand cached data will be used."\
            .format(filepath))
    else:
        file_id = "1rS_RfcoiSVyCpRTfesX5Nn6DQliFb4DR"
        print("Downloading AutoDL data from Google Drive...")
        download_file_from_google_drive(file_id, filepath)
        print("Data successfully downloaded to {}.".format(filepath))


def parse_autodl_data(filepath=None, save=False):
    if filepath is None:
        pwd = os.path.dirname(__file__)
        filepath = os.path.join(pwd, os.pardir, 
            'datasets', 'AutoDL', 'all_results.csv')
    df = pd.read_csv(filepath, index_col=0)
    df_final = df[df['phase'].isin(['final', 'feedback'])]
    participant_names = sorted(list(df_final['participant_name'].unique()))
    task_names = list(df_final['task_name'].unique())
    df_final = df_final.set_index(keys=['participant_name', 'task_name'])
    D = len(task_names)
    A = len(participant_names)
    perfs = np.zeros(shape=(D, A))
    for a, pn in enumerate(participant_names):
        for d, tn in enumerate(task_names):
            alc_score = df_final.loc[pn, tn]['alc_score']
            perfs[d][a] = alc_score
    datasets = task_names
    algos = participant_names
    # name='AutoDL-{}'.format(phase)
    name = 'AutoDL'
    da_matrix = DAMatrix(perfs=perfs, datasets=datasets, 
                         algos=algos, name=name)
    if save:
        da_matrix.save(path_to_dir=os.path.dirname(filepath))
    return da_matrix


def parse_cepairs_data(filepath=None, save=False):
    if filepath is None:
        filepath = os.path.join(ROOT_DIR, os.pardir, 
            'datasets', 'CEpairs', 'error_BER_withoutNa.csv')
    df = pd.read_csv(filepath, index_col=0)
    algos = [BetaAlgo(name=name_algo) for name_algo in df.columns]
    datasets = [BetaDataset(name=str(idx_dataset)) for idx_dataset in df.index]
    perfs = np.array(df)
    name = 'CEpairs'
    da_matrix = DAMatrix(perfs=perfs, datasets=datasets,
                         algos=algos, name=name)
    if save:
        da_matrix.save(path_to_dir=os.path.dirname(filepath))
    return da_matrix


def plot_error_bars_empirical_vs_theoretical():
    da_matrix = parse_autodl_data()
    perfs = da_matrix.perfs
    n_T = len(da_matrix.datasets)
    n_B = len(da_matrix.algos)
    delta = 0.05
    def get_error_bar(n_T, n_B, delta):
        error_bar = np.sqrt((np.log(n_B) + np.log(2 / delta)) / (2 * n_T))
        return error_bar
    meta_train = []
    meta_test = []
    ebs_emp = []
    ebs_the = []
    algo_names = da_matrix.algos
    print(algo_names)
    for i_a in range(n_B):
        alcs = perfs[:, i_a]
        perf_meta_train = np.mean(alcs[:5])
        perf_meta_test = np.mean(alcs[5:])
        eb_emp = perf_meta_test - perf_meta_train
        
        meta_train.append(perf_meta_train)
        meta_test.append(perf_meta_test)
        ebs_emp.append(eb_emp)

        eb_the = get_error_bar(n_T, i_a + 1, delta)
        ebs_the.append(eb_the)
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    df = pd.DataFrame(
        {
            'algo_name': algo_names,
            'meta_train': meta_train,
            'meta_test': meta_test,
            'ebs_emp': ebs_emp,
        }
    )
    df = df.sort_values(by='meta_train', ascending=False)
    print(df)

    meta_train = list(df['meta_train'])
    meta_test = list(df['meta_test'])
    ebs_emp = list(df['ebs_emp'])
    algo_names = list(df['algo_name'])

    ebs_emp_max = []
    eb_emp_max = - 2 ** 32
    for i_a in range(n_B):
        eb_emp = ebs_emp[i_a]
        eb_emp_max = max(abs(eb_emp), eb_emp_max)
        ebs_emp_max.append(eb_emp_max)

    ax.plot(meta_train, label='meta-train')
    ax.plot(meta_test, label='meta-test')
    ax.plot(ebs_emp, label='meta-test - meta-train')
    ax.plot(ebs_emp_max, label='max(abs(meta-test - meta-train))')
    ax.plot(ebs_the, label='Theoretical error bars (delta={:.2f})'.format(delta))

    ax.set_xticks(range(n_B))
    ax.set_xticklabels(algo_names, rotation=45)

    plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def to_df_for_cd_diagram(da_matrix):
    classifier_names = []
    dataset_names = []
    accuracies = []

    datasets = da_matrix.datasets
    algos = da_matrix.algos

    for i_d in range(len(datasets)):
        for i_a in range(len(algos)):
            classifier_names.append(algos[i_a].name)
            dataset_names.append(datasets[i_d].name)
            accuracies.append(da_matrix.perfs[i_d, i_a])

    df = pd.DataFrame({
        'classifier_name': classifier_names,
        'dataset_name': dataset_names,
        'accuracy': accuracies,
    })

    return df