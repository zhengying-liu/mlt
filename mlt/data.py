# Author: Zhengying Liu
# Creation date: 4 Dec 2020

import json
import numpy as np
import os

np.random.seed(42)


class DAMatrix(object):
    """A DAMatrix object contains a performance matrix, given a list of datasets
    and a list of algorithms. Each row corresponds to a dataset and each column
    corresponds to an algorithm. Each entry (which could be `None`) of i-th row
    and j-th column corresponds to the performance of executing j-th algorithm
    on i-th dataset.
    """

    def __init__(self, perfs=None, datasets=None, algos=None, name=None):
        """
        Args:
          perfs: a 2-D NumPy array of shape (`n_datasets`, `n_algos`)
          datasets: a list of BetaDataset objects
          algos: a list of BetaAlgo objects
          name: str, name of the DAMatrix
        """
        self.datasets = datasets if datasets else []
        self.algos = algos if algos else []
        self.name = str(name)

        # Initialize empty performance matrix if not provided
        if perfs is None:
            n_datasets = len(self.datasets)
            n_algos = len(self.algos)
            shape = (n_datasets, n_algos)
            self.perfs = np.full(shape, np.nan)
        else:
            self.perfs = perfs

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
        perfs_filename = "{}.perfs".format(self.name)
        perfs_path = os.path.join(path_to_dir, perfs_filename)
        np.savetxt(perfs_path, self.perfs)

        # Save list of datasets
        datasets_filename = "{}.datasets".format(self.name)
        datasets_path = os.path.join(path_to_dir, datasets_filename)
        with open(datasets_path, 'w') as f:
            f.write(str(self.datasets))

        # Save list of algos
        algos_filename = "{}.algos".format(self.name)
        algos_path = os.path.join(path_to_dir, algos_filename)
        with open(algos_path, 'w') as f:
            f.write(str(self.algos))

        print("Successfully saved DAMatrix {} to {}."\
            .format(self.name, path_to_dir))

        return path_to_dir

    @classmethod
    def load(cls, path_to_dir):
        # Load DAMatrix
        files = os.listdir(path_to_dir)
        ext_name = '.perfs'
        perfs_files = [x for x in files if x.endswith(ext_name)]
        if len(perfs_files) == 0:
            raise FileNotFoundError("No `.perfs` file no in {}".format(path_to_dir))
        elif len(perfs_files) > 1:
            raise FileNotFoundError("Multiple `.perfs` files found in {}: {}"\
                .format(path_to_dir, perfs_files))
        
        perfs_file = perfs_files[0]
        name = perfs_file[:-len(ext_name)]
        perfs_path = os.path.join(path_to_dir, perfs_file)
        perfs = np.loadtxt(perfs_path)

        # TODO: load datasets and algos

        return DAMatrix(perfs=perfs, name=name)
        

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

    def __init__(self, n_datasets=1000, name='Case3d_DAMatrix'):
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


class BetaAlgo(object):

    def __init__(self, name=None, code=None):
        self.name = name
        self.code = code

    def run(self, dataset):
        pass

    def __repr__(self):
        return "BetaAlgo(name={}, code={})".format(self.name, self.code)


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