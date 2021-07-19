# Author: Zhengying Liu
# Creation date: 4 Dec 2020

from mlt.meta_learner import RandomSearchMetaLearner
from mlt.meta_learner import MeanMetaLearner
from mlt.meta_learner import GreedyMetaLearner
from mlt.meta_learner import OptimalMetaLearner
from mlt.meta_learner import TopkRankMetaLearner
from mlt.meta_learner import run_and_plot_learning_curve, run_leave_one_out
from mlt.meta_learner import run_meta_validation
from mlt.meta_learner import plot_meta_learner_with_different_ranks
from mlt.meta_learner import plot_meta_learner_with_different_true_ranks
from mlt.meta_learner import plot_alc_vs_rank
from mlt.meta_learner import binarize
from mlt.meta_learner import save_perfs
from mlt.meta_learner import get_the_meta_learners
from mlt.meta_learner import generate_binary_matrix_with_rank
from mlt.meta_learner import run_once_random
from mlt.meta_learner import plot_meta_learner_with_different_cardinal_clique
from mlt.meta_learner import plot_alc_vs_cardinal_clique
from mlt.meta_learner import get_conditional_prob
from mlt.meta_learner import plot_error_bar_vs_B
from mlt.meta_learner import get_average_rank

from mlt.figures import get_meta_scores_vs_n_tasks
from mlt.figures import plot_curve_with_error_bars

from mlt.data import DAMatrix, NFLDAMatrix, Case2DAMatrix, Case3dDAMatrix
from mlt.data import BinarizedMultivariateGaussianDAMatrix
from mlt.data import ComplementaryDAMatrix, CopulaCliqueDAMatrix
from mlt.data import parse_autodl_data
from mlt.data import get_da_matrix_from_real_dataset_dir

from mlt.utils import save_fig

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy.optimize import minimize, LinearConstraint
try:
    from cvxopt import matrix, solvers
    USE_CVXOPT = True
except:
    USE_CVXOPT = False


def test_run_and_plot_learning_curve():
    da_matrix = NFLDAMatrix()
    rs_meta_learner = RandomSearchMetaLearner()
    run_and_plot_learning_curve([rs_meta_learner], da_matrix)


def test_mean_meta_learner():
    n_algos = 13
    thetas = np.arange(n_algos) / n_algos
    # da_matrix = NFLDAMatrix()
    case2_da_matrix = Case2DAMatrix(thetas=thetas)
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    meta_learners = [rs_meta_learner, mean_meta_learner]
    run_and_plot_learning_curve(meta_learners, case2_da_matrix)


def test_all_meta_learners():
    n_algos = 5
    thetas = np.arange(n_algos) / n_algos / 10
    print("thetas:", thetas)
    # da_matrix = NFLDAMatrix()
    case2_da_matrix = Case2DAMatrix(thetas=thetas)
    i_dataset = -2
    print("True perfs:", case2_da_matrix.perfs[i_dataset])
    # print(case2_da_matrix.perfs)
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    greedy_meta_learner = GreedyMetaLearner()
    optimal_meta_learner = OptimalMetaLearner()
    meta_learners = [rs_meta_learner, 
                     mean_meta_learner, 
                     greedy_meta_learner, 
                     optimal_meta_learner]
    run_and_plot_learning_curve(meta_learners, case2_da_matrix, 
                                i_dataset=i_dataset)


def test_case3d_damatrix():
    da_matrix = Case3dDAMatrix()
    i_dataset = -1
    print("True perfs:", da_matrix.perfs[i_dataset])
    # print(case2_da_matrix.perfs)
    rs_meta_learner = RandomSearchMetaLearner()
    mean_meta_learner = MeanMetaLearner()
    greedy_meta_learner = GreedyMetaLearner()
    optimal_meta_learner = OptimalMetaLearner()
    meta_learners = [rs_meta_learner, 
                     mean_meta_learner, 
                     greedy_meta_learner, 
                     optimal_meta_learner]
    run_and_plot_learning_curve(meta_learners, da_matrix, 
                                i_dataset=i_dataset)


def test_damatrix(ClsDAMatrix, kwargs=None, i_dataset=-1,
                  meta_learners=None, include_optimal=False):
    if kwargs is None:
        kwargs = {}

    da_matrix = ClsDAMatrix(**kwargs)

    print("Performance matrix:", da_matrix.perfs)

    print("True perfs:", da_matrix.perfs[i_dataset])

    if meta_learners is None:
        rs_meta_learner = RandomSearchMetaLearner()
        mean_meta_learner = MeanMetaLearner()
        greedy_meta_learner = GreedyMetaLearner()
        optimal_meta_learner = OptimalMetaLearner()
        meta_learners = [rs_meta_learner, 
                        mean_meta_learner, 
                        greedy_meta_learner,
                        ]
        if include_optimal:
            meta_learners.append(optimal_meta_learner)

    run_and_plot_learning_curve(meta_learners, da_matrix, 
                                i_dataset=i_dataset)


def test_leave_one_out(ClsDAMatrix, kwargs=None,
                  meta_learners=None, include_optimal=False):
    if kwargs is None:
        kwargs = {}

    da_matrix = ClsDAMatrix(**kwargs)

    print("Performance matrix:")
    print(da_matrix.perfs)

    if meta_learners is None:
        rs_meta_learner = RandomSearchMetaLearner()
        mean_meta_learner = MeanMetaLearner()
        greedy_meta_learner = GreedyMetaLearner()
        optimal_meta_learner = OptimalMetaLearner()
        meta_learners = [rs_meta_learner, 
                        mean_meta_learner, 
                        greedy_meta_learner,
                        ]
        if include_optimal:
            meta_learners.append(optimal_meta_learner)

    fig = run_leave_one_out(meta_learners, da_matrix, n_runs=10)

    # Save results
    np.savetxt('perfs.npy', da_matrix.perfs.astype(int), fmt='%i')
    fig.savefig('result.jpg')


def test_run_meta_validation():
    da_matrix = Case3dDAMatrix(n_datasets=2000)
    meta_learners = get_the_meta_learners()
    run_meta_validation(meta_learners, da_matrix)


def run_expe(da_matrix, meta_learners, 
             name_expe=None,
             results_dir='../results', with_once_random=False, ylim=None, 
             show_legend=False, figsize=(5,3)):
    """Use run_meta_validation to run experiment."""
    if with_once_random:
        fig = run_once_random(da_matrix)
    else:
        fig = None

    fig = run_meta_validation(meta_learners, da_matrix, fig=fig, ylim=ylim, 
                              show_legend=show_legend, figsize=figsize)

    # Create directory for the experiment
    expe_dir = os.path.join(results_dir, str(name_expe))
    os.makedirs(expe_dir, exist_ok=True)

    # Save performance matrix and the figure
    save_perfs(da_matrix.perfs.astype(int), name_expe=name_expe)
    save_fig(fig, name_expe=name_expe)


def run_nfl():
    n_datasets = 20000
    n_algos = 5
    perfs = (np.random.rand(n_datasets, n_algos) < 0.5).astype(int)
    name_expe = 'nfl'
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3a():
    n_datasets = 20000
    n_algos = 4
    col = (np.random.rand(n_datasets, 1) < 0.5).astype(int)
    perfs = np.concatenate([col] * n_algos, axis=1)
    name_expe = '3a-repeated-columns'
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe, ylim=(0.45, 1.05))


def run_3b():
    n_datasets = 20000
    n_algos = 5
    X1 = (np.random.rand(n_datasets, 1) < 0.5).astype(int)
    X2 = 1 - X1
    perfs = np.concatenate([X1, X1, X2, X2], axis=1)
    name_expe = '3b-complementary-2-algos'
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    # da_matrix = ComplementaryDAMatrix()
    meta_learners = get_the_meta_learners(exclude_greedy_plus=True)
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3d():
    n_datasets = 20000
    name_expe = '3d'
    da_matrix = Case3dDAMatrix(n_datasets=n_datasets, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def get_multivariate_bernoulli_3f(epsilon=1e-1, n_datasets=20000, 
                                  use_cvxopt=USE_CVXOPT):
    """ 
        ABCD
    x0: 0000
    x1: 0001
    x2: 0010
    x3: 0011
    x4: 0100
    x5: 0101
    x6: 0110
    x7: 0111
    x8: 1000
    x9: 1001
    x10: 1010
    x11: 1011
    x12: 1100
    x13: 1101
    x14: 1110
    x15: 1111
    """
    n_algos = 4
    e = epsilon

    B = [
        0.5 - 2 * e,    # P(A=0)
        0.5 - e,        # P(B=0)
        0.5 + e,        # P(C=0)
        0.5 + 2 * e,    # P(D=0)
        0,              # P(B=0|A=0)        = 0.5 + 2e
        0,              # P(C=0|A=0)        = 0.5 + 2e
        0,              # P(D=0|A=0)        = 0.5 + e
        0,              # P(C=0|A=0,B=0)    = 0
        0,              # P(B=0|A=0,D=0)    = 0.5
        0,              # P(C=0|A=0,D=0)    = 0.5
        1,              # P(all)            = 1
    ]
    
    A = np.zeros(shape=(len(B), 2 ** n_algos))
    indicess = [[] for _ in range(n_algos)]

    # A=0
    indices_A0 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = "0{}{}{}".format(i, j, k)
                idx = int(s, base=2)
                indices_A0.append(idx)
    # B=0
    indices_B0 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = "{}0{}{}".format(i, j, k)
                idx = int(s, base=2)
                indices_B0.append(idx)
    # C=0
    indices_C0 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = "{}{}0{}".format(i, j, k)
                idx = int(s, base=2)
                indices_C0.append(idx)
    # D=0
    indices_D0 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = "{}{}{}0".format(i, j, k)
                idx = int(s, base=2)
                indices_D0.append(idx)
    # A=0,B=0
    indices_A0B0 = []
    for i in range(2):
        for j in range(2):
            s = "00{}{}".format(i, j)
            idx = int(s, base=2)
            indices_A0B0.append(idx)
    # A=0,C=0
    indices_A0C0 = []
    for i in range(2):
        for j in range(2):
            s = "0{}0{}".format(i, j)
            idx = int(s, base=2)
            indices_A0C0.append(idx)
    # A=0,D=0
    indices_A0D0 = []
    for i in range(2):
        for j in range(2):
            s = "0{}{}0".format(i, j)
            idx = int(s, base=2)
            indices_A0D0.append(idx)
    # A=0,B=0,C=0
    indices_A0B0C0 = []
    for i in range(2):
        s = "000{}".format(i)
        idx = int(s, base=2)
        indices_A0B0C0.append(idx)
    # A=0,C=0,D=0
    indices_A0C0D0 = []
    for i in range(2):
        s = "0{}00".format(i)
        idx = int(s, base=2)
        indices_A0C0D0.append(idx)
    # A=0,B=0,D=0
    indices_A0B0D0 = []
    for i in range(2):
        s = "00{}0".format(i)
        idx = int(s, base=2)
        indices_A0B0D0.append(idx)
    
    for idx in indices_A0:
        A[0, idx] += 1      # P(A)
    for idx in indices_B0:
        A[1, idx] += 1      # P(B)
    for idx in indices_C0:
        A[2, idx] += 1      # P(C)
    for idx in indices_D0:
        A[3, idx] += 1      # P(D)

    factor_importance = 1
    fi = factor_importance
    
    # P(B=0|A=0) = 0.5 + 2e
    for idx in indices_A0B0:
        A[4, idx] += 1
    for idx in indices_A0:
        A[4, idx] += - (0.5 + 2 * e)
    A[4] *= fi ** 2

    # P(C=0|A=0) = 0.5 + 2e
    for idx in indices_A0C0:
        A[5, idx] += 1
    for idx in indices_A0:
        A[5, idx] += - (0.5 + 2 * e)
    A[5] *= fi ** 2

    # We want Greedy to choose D at step 2
    # P(D=0|A=0) = 0.5 + e
    for idx in indices_A0D0:
        A[6, idx] += 1
    for idx in indices_A0:
        # A[6, idx] += - (0.5 + e)
        A[6, idx] += - 0.5
    A[6] *= fi

    # We want Mean to be perfect at step 3
    # P(C=0|A=0,B=0) = 0
    for idx in indices_A0B0C0:
        A[7, idx] += 1
    A[7] *= fi

    # We wang Greedy to be bad at step 3
    # P(C=0|A=0,D=0) = 0.5
    for idx in indices_A0C0D0:
        A[8, idx] += 1
    for idx in indices_A0D0:
        A[8, idx] += - 0.5
    A[8] *= fi
    # P(B=0|A=0,D=0) = 0.5
    for idx in indices_A0B0D0:
        A[9, idx] += 1
    for idx in indices_A0D0:
        A[9, idx] += - 0.5
    A[9] *= fi

    # P(all) = 1
    for idx in range(2 ** n_algos):
        A[10, idx] += 1

    if not use_cvxopt:
        # # Use optimization tool to solve the equation
        def f(x):
            y = np.dot(A, x) - B
            return np.dot(y, y)

        cons = [
            {'type': 'eq', 'fun': lambda x: x.sum() - 1},
            LinearConstraint(
                A=np.eye(2 ** n_algos), 
                lb=np.zeros(2 ** n_algos),
                ub=np.ones(2 ** n_algos)
            ),
        ]
        res = minimize(f, np.zeros(2 ** n_algos), method='SLSQP', constraints=cons, 
                                options={'disp': False})

        x = np.array(res['x'])

    else:
        # Use CVXOPT
        print(A)
        print(A.shape)
        print("np.linalg.matrix_rank(A)", np.linalg.matrix_rank(A))
        
        P = matrix(A.T.dot(A))
        q = matrix(-A.T.dot(B))
        G = np.concatenate([np.eye(16), -np.eye(16)])
        G = matrix(G)
        h = [1.0] * 16 + [0.0] * 16
        h = matrix(h)
        # AA = np.array([1.0] * 16).reshape(1, 16)
        # AA = matrix(AA)
        # b = matrix(1.0)
        # sol = solvers.qp(P, q, G, h, AA, b)
        sol = solvers.qp(P, q, G, h)
        x = np.array(sol['x']).reshape(16)

    x = [e if e >=0 else 0 for e in x]
    x = np.array(x)
    x = x / x.sum()

    residu = A.dot(x) - B
    print("Ax:", A.dot(x))
    print("B:", B)
    print("x.sum()", x.sum())
    print(x >= 0)
    print(x <= 1)
    print("residu.shape:", residu.shape)
    print("residu:", residu)
    print("residu norm:", residu.dot(residu))
    print("x:", x)

    perfs = []
    for i in range(n_datasets):
        idx = np.random.choice(2 ** n_algos, p=x)
        bits = []
        for _ in range(n_algos):
            bits.append(idx % 2)
            idx //= 2
        bits = bits[::-1]
        perfs.append(bits)
    perfs = np.array(perfs)

    name = '3f'
    da_matrix = DAMatrix(perfs=perfs, name=name)

    PA0 = sum([x[i] for i in indices_A0])
    PA0C0 = sum([x[i] for i in indices_A0C0])
    print("Real P(C=0|A=0)={}".format(PA0C0 / PA0))
    PA0 = sum([x[i] for i in indices_A0])
    PA0D0 = sum([x[i] for i in indices_A0D0])
    print("Real P(D=0|A=0)={}".format(PA0D0 / PA0))

    return da_matrix


def test_get_multivariate_bernoulli_3f():
    da_matrix = get_multivariate_bernoulli_3f()
    perfs = da_matrix.perfs
    df = pd.DataFrame(perfs)

    df_A0 = df[df[0] == 0]
    df_B0 = df[df[1] == 0]
    df_C0 = df[df[2] == 0]
    df_D0 = df[df[3] == 0]

    PA0 = len(df_A0) / len(df)
    print("P(A=0)={}".format(PA0))
    PB0 = len(df_B0) / len(df)
    print("P(B=0)={}".format(PB0))
    PC0 = len(df_C0) / len(df)
    print("P(C=0)={}".format(PC0))
    PD0 = len(df_D0) / len(df)
    print("P(D=0)={}".format(PD0))

    df_A0B0 = df_A0[df_A0[1] == 0]
    df_A0C0 = df_A0[df_A0[2] == 0]
    df_A0D0 = df_A0[df_A0[3] == 0]
    PA0B0 = len(df_A0B0) / len(df_A0)
    print("P(B=0|A=0)={}".format(PA0B0))
    PA0C0 = len(df_A0C0) / len(df_A0)
    print("P(C=0|A=0)={}".format(PA0C0))
    PA0D0 = len(df_A0D0) / len(df_A0)
    print("P(D=0|A=0)={}".format(PA0D0))

    df_A0B0C0 = df_A0B0[df_A0B0[2] == 0]
    df_A0D0C0 = df_A0D0[df_A0D0[2] == 0]
    df_A0D0B0 = df_A0D0[df_A0D0[1] == 0]
    PA0B0C0 = len(df_A0B0C0) / len(df_A0B0)
    print("P(C=0|A=0,B=0)={}".format(PA0B0C0))
    PA0D0C0 = len(df_A0D0C0) / len(df_A0B0)
    print("P(C=0|A=0,D=0)={}".format(PA0D0C0))
    PA0D0B0 = len(df_A0D0B0) / len(df_A0B0)
    print("P(B=0|A=0,D=0)={}".format(PA0D0B0))

    print("P(D=0|A=0) < P(B=0|A=0):", PA0D0 < PA0B0)
    print("P(D=0|A=0) < P(C=0|A=0):", PA0D0 < PA0C0)


def get_da_matrix_3f():
    fpath = '../results/da_matrix_4f.txt'
    perfs = np.loadtxt(fpath)
    da_matrix = DAMatrix(perfs=perfs)
    return da_matrix
    

def run_3f():
    da_matrix = get_multivariate_bernoulli_3f()
    # da_matrix = get_da_matrix_3f()
    name_expe = '3f'
    meta_learners = get_the_meta_learners()
    print(da_matrix.perfs.shape)
    print(da_matrix.perfs.mean(axis=0))
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3f_old():
    n_datasets = 20000
    name_expe = '3f'
    epsilon = 1e-1
    A = (np.random.rand(n_datasets * 2, 1) < 0.5 + 2 * epsilon).astype(int)
    B = (np.random.rand(n_datasets * 2, 1) < 0.5 +  epsilon).astype(int)
    C = (np.random.rand(n_datasets * 2, 1) < 0.5 -  epsilon).astype(int)
    D = (np.random.rand(n_datasets * 2, 1) < 0.5 -  2 * epsilon).astype(int)
    perfs = np.concatenate([A, B, C, D], axis=1)

    valid_rows = []
    for row in perfs:
        if not (row[0] == 0 and row[1] == 0 and row[2] == 0):
            valid_rows.append(row)
        if len(valid_rows) == n_datasets:
            break
    perfs = np.array(valid_rows)[:n_datasets]
    assert len(perfs) == n_datasets
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners()
    run_expe(da_matrix, meta_learners, name_expe=name_expe)


def run_3g():
    n_datasets = 20000
    name_expe = '3g'
    epsilon = 1e-1
    X1 = (np.random.rand(n_datasets, 1) < 0.5 + epsilon).astype(int)
    X2 = 1 - X1
    perfs = np.concatenate([X1, X1, X1, X2, X2, X2], axis=1)
    da_matrix = DAMatrix(perfs=perfs, name=name_expe)
    meta_learners = get_the_meta_learners(exclude_greedy_plus=True)
    run_expe(da_matrix, meta_learners, name_expe=name_expe, show_legend=False)


def test_binarize():
    matrix = np.random.rand(10, 3)
    bm = binarize(matrix)
    print(matrix)
    print(bm)


def test_generate_binary_matrix_with_rank():
    for m in range(1, 10):
        for n in range(1, 10):
            for rank in range(min(m, n) + 1):
                matrix = generate_binary_matrix_with_rank(rank, m, n)
                print(matrix)
                print(rank)
                assert np.linalg.matrix_rank(matrix) == rank


def run_on_real_dataset(dataset_dir):
    """`dataset_dir` should contain a file `*.data` in NumPy format."""
    if os.path.isdir(dataset_dir):
        data_files = [x for x in os.listdir(dataset_dir) 
                        if x.endswith('.data')]
        if len(data_files) != 1:
            raise ValueError("The dataset directory {} ".format(dataset_dir) + 
                                "should contain one `.data` file but got " +
                                "{}.".format(data_files))
        data_file = data_files[0]
        data_path = os.path.join(dataset_dir, data_file)
        name_expe = data_file.split('.')[0]

        # Load real dataset and binarize
        perfs = binarize(np.loadtxt(data_path))
        # da_matrix = DAMatrix(perfs=perf, name=name_expe)
        da_matrix = DAMatrix.load(dataset_dir)
        da_matrix.perfs = perfs
        meta_learners = get_the_meta_learners(exclude_optimal=True)[1:] # Exclude random

        run_expe(da_matrix, meta_learners=meta_learners, 
                name_expe=name_expe, with_once_random=True, show_legend=True)
    else:
        raise ValueError("Not a directory: {}".format(dataset_dir))


def run_on_all_real_datasets(datasets_dir=None):
    if datasets_dir is None:
        datasets_dir = "../datasets"
    
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            run_on_real_dataset(dataset_dir)


def get_real_datasets_dataset_dirs(datasets_dir=None):
    if datasets_dir is None:
        datasets_dir = "../datasets"
    
    dataset_dirs = []
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            dataset_dirs.append(dataset_dir)
    return dataset_dirs


def test_run_once_random():
    da_matrix = Case3dDAMatrix(n_datasets=20000)
    run_once_random(da_matrix, n_meta_learners=10)


def run_leave_one_out_on_real_datasets():
    dataset_dirs = get_real_datasets_dataset_dirs()
    meta_learners = get_the_meta_learners(exclude_optimal=True)[1:]
    for dataset_dir in dataset_dirs:
        name_expe = "LOO-{}".format(os.path.basename(dataset_dir))
        da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
        fig = run_once_random(da_matrix, leave_one_out=True)
        fig = run_leave_one_out(meta_learners, da_matrix, use_all=True, fig=fig)
        save_fig(fig, name_expe=name_expe)
        save_perfs(da_matrix.perfs, name_expe=name_expe)


def test_get_conditional_prob():
    perfs = (np.random.rand(100, 5) < 0.5).astype(int)
    cp = get_conditional_prob(perfs, cond_cols=[0])
    cp = get_conditional_prob(perfs, cond_cols=[0], i_target=1)
    print(cp)


def test_plot_error_bar_vs_B():
    plot_error_bar_vs_B()
    plt.show()


def test_get_meta_scores_vs_n_tasks():
    da_matrix = parse_autodl_data()
    meta_learner = MeanMetaLearner()
    curves = get_meta_scores_vs_n_tasks(da_matrix, meta_learner)
    # print(curves)
    return curves


def test_unit_TopkRankMetaLearner():
    meta_learner = TopkRankMetaLearner()

    for _ in range(10):
        perfs = np.array([
            [1, 3, 2],
            [4, 6, 5],
        ])
        n_algos = perfs.shape[1]
        perm = np.random.permutation(n_algos)
        perfs = perfs[:, perm]
        da_matrix = DAMatrix(perfs=perfs)

        meta_learner.meta_fit(da_matrix)
        idx = meta_learner.indices_algo_to_reveal
        # print(idx)
        # print([da_matrix.algos[i] for i in idx])
        # print(perfs, idx[0], perm)
        assert perm[idx[0]] == 1



def test_TopkRankMetaLearner():
    meta_learner = TopkRankMetaLearner()
    
    datasets_dir="../datasets"
    dataset_names = [
        # 'URV-10-200-20', 
        # 'artificial_r50c20r20', 
        'AutoDL', 'AutoML', 'OpenML-Alors', 'Statlog']
    ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    for d in ds:
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            n_datasets = len(da_matrix.datasets)
            print("Meta-dataset:", da_matrix.name)
            print("n_datasets:", n_datasets)
            excluded_indices = range(n_datasets // 2)
            print("Number of datasets used for validation:", len(excluded_indices))
            meta_learner.meta_fit(da_matrix, plot=True, 
                excluded_indices=excluded_indices)


def test_get_average_rank():
    perfs = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    avg_rank = get_average_rank(perfs)
    assert np.all(np.isclose(avg_rank, np.array([2, 1, 0])))
    avg_rank = get_average_rank(perfs, negative_score=True)
    assert np.all(np.isclose(avg_rank, np.array([0, 1, 2])))

    perfs = np.array([
        [],
    ])
    avg_rank = get_average_rank(perfs)
    assert len(avg_rank) == 0


if __name__ == '__main__':
    pass
    # test_run_and_plot_learning_curve()
    # test_mean_meta_learner()
    # test_all_meta_learners()
    # test_case3d_damatrix()
    # test_run_meta_validation()
    # test_binarize()
    # test_generate_binary_matrix_with_rank()
    # test_get_multivariate_bernoulli_3f()
    # test_get_conditional_prob()

    # run_3a()
    # run_3b()
    # run_3d()
    # run_3f()
    # run_3g()
    # run_nfl()
    
    # plot_meta_learner_with_different_ranks()
    # plot_meta_learner_with_different_true_ranks()
    # plot_alc_vs_rank()

    # run_on_all_real_datasets()
    # run_on_real_dataset("../datasets/AutoDL")

    # run_leave_one_out_on_real_datasets()

    # plot_meta_learner_with_different_cardinal_clique()

    # plot_alc_vs_cardinal_clique()

    # get_multivariate_bernoulli_3f()

    # test_plot_error_bar_vs_B()

    # test_get_meta_scores_vs_n_tasks()

    # test_plot_curve_with_error_bars()

    test_TopkRankMetaLearner()

    # test_get_average_rank()

    # test_unit_TopkRankMetaLearner()