# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.utils import download_file_from_google_drive

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


def get_da_matrix_from_real_dataset_dir(dataset_dir):
    if os.path.isdir(dataset_dir):
        da_matrix = DAMatrix.load(dataset_dir)
        return da_matrix
    else:
        raise ValueError("Not a directory: {}".format(dataset_dir))


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