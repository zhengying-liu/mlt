# Author: Zhengying LIU
# Create: 6 May 2021

from mlt.data import DAMatrix
from mlt.data import get_da_matrix_from_real_dataset_dir
from mlt.meta_learner import MeanMetaLearner
from mlt.utils import save_fig
from mlt.utils import get_theoretical_error_bar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_curve_with_error_bars(li_mean, li_std, fig=None, label=None, **kwargs):
    if fig is None:
        fig = plt.figure()

    if len(fig.axes) > 0:
        ax = fig.axes[0]
    else:
        ax = fig.add_subplot(1, 1, 1)

    # Integer x-axis ticks
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    a_mean = np.array(li_mean)
    a_std = np.array(li_std)
    upper = a_mean + a_std
    lower = a_mean - a_std

    X = np.arange(len(li_mean)) + 1
    
    ax.plot(X, li_mean, label=label, **kwargs)

    ax.fill_between(X, upper, lower, alpha=0.3)
        
    return fig


############################
## ALC vs number of tasks ##
############################
def get_meta_scores_vs_n_tasks(da_matrix, meta_learner, 
                               n_meta_train=5,
                               repeat=100):
    """Get meta-scores (meta-train, meta-valid, meta-test) vs number of tasks
    in the meta-training set. This gives a sort of (meta-)learning curves.

    Suppose there are in total `T` tasks in meta-train. At step `t`, choose 
    randomly `t` tasks among the `T` tasks and apply the meta-learner. Use 
    the `T - t` tasks for meta-validation and use meta-test for test score. 
    Repeat this process `repeat` times and compute the mean and std.

    Here we only use the first algorithm predicted by the meta-learner.

    N.B. For a DA matrix, we suppose the first `n_meta_train` tasks are used 
    as meta-train and the rest is used as  meta-test.
    """
    n_datasets = len(da_matrix.datasets)
    if n_meta_train > n_datasets:
        raise ValueError("The number of meta-train tasks should be less than " +
                         "or equal to the total number of tasks." +
                         "But got {} > {}.".format(n_meta_train, n_datasets))
    T = n_meta_train

    da_meta_train, da_meta_test = da_matrix.train_test_split(
        train_size=n_meta_train, shuffling=False
    )

    mean_tr = []
    std_tr = []
    mean_va = []
    std_va = []
    mean_te = []
    std_te = []

    for t in range(1, T + 1):
        s_tr = []
        s_va = []
        s_te = []
        for _ in range(repeat):
            # Choose t among T tasks for meta-train, without replacement
            valid_indices = set(np.random.choice(T, T - t, replace=False))
            meta_learner.meta_fit(da_meta_train, valid_indices)
            i_algo = meta_learner.indices_algo_to_reveal[0]
            # print(da_meta_train.algos[i_algo])
            # print(valid_indices)

            # Meta-train & meta-valid score
            sum_tr = 0
            sum_va = 0
            for i in range(T):
                if i in valid_indices:
                    sum_va += da_meta_train.perfs[i, i_algo]
                else:
                    sum_tr += da_meta_train.perfs[i, i_algo]
            avg_tr = sum_tr / t
            avg_va = sum_va / (T - t) if T > t else np.nan
            s_tr.append(avg_tr)
            s_va.append(avg_va)

            # Meta-test score
            avg_te = np.mean(da_meta_test.perfs[:, i_algo])
            s_te.append(avg_te)
        
        mean_tr.append(np.mean(s_tr))
        std_tr.append(np.std(s_tr))
        mean_va.append(np.mean(s_va))
        std_va.append(np.std(s_va))
        mean_te.append(np.mean(s_te))
        std_te.append(np.std(s_te))
    
    mean_tr = np.array(mean_tr)
    std_tr = np.array(std_tr)
    mean_va = np.array(mean_va)
    std_va = np.array(std_va)
    mean_te = np.array(mean_te)
    std_te = np.array(std_te)

    return mean_tr, std_tr, mean_va, std_va, mean_te, std_te


def plot_score_vs_n_tasks_with_error_bars(repeat=100):
    datasets_dir = "../datasets"

    score_names = {
        'artificial_r50c20r20': 'Performance',
        'AutoDL': 'ALC',
        'AutoML': 'BAC or R2',
        'OpenML-Alors': 'Accuracy',
        'Statlog': 'Error rate',
    }
    
    for d in os.listdir(datasets_dir):
        dataset_dir = os.path.join(datasets_dir, d)
        if os.path.isdir(dataset_dir):
            da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
            meta_learner = MeanMetaLearner()
            

            name_expe = "alc-vs-n_tasks"
            if d == 'AutoDL':
                n_meta_train = 5
            else:
                n_meta_train = da_matrix.perfs.shape[0] // 2
            
            n_meta_test = da_matrix.perfs.shape[0] - n_meta_train

            curves = get_meta_scores_vs_n_tasks(da_matrix, meta_learner, 
                n_meta_train=n_meta_train)

            score_name = score_names[d]

            fig = plot_curve_with_error_bars(curves[0], curves[1], 
                label='meta-train')
            fig = plot_curve_with_error_bars(curves[2], curves[3], fig=fig, 
                label='meta-valid')
            fig = plot_curve_with_error_bars(curves[4], curves[5], fig=fig, 
                label='meta-test')
            plt.xlabel("Number of tasks used for meta-training " +
                "(|Dtr|={}, |Dte|={})".format(n_meta_train, n_meta_test))
            plt.ylabel("Average {} score".format(score_name))
            plt.legend()
            plt.title("{} - {} VS #tasks".format(d, score_name))
            plt.show()
            save_fig(fig, name_expe=name_expe, 
                filename="{}-alc-vs-n_tasks.jpg".format(d))


#################################
## ALC vs number of algorithms ##
#################################
def get_meta_scores_vs_n_algos(da_matrix, meta_learner, 
                               n_meta_train=5,
                               repeat=100):
    """Get meta-scores (meta-train, meta-valid, meta-test) vs number of 
    algorithms in the meta-training set. This gives (meta-)learning curves.

    Suppose there are in total `A` tasks in meta-train. At step `a`, choose 
    randomly `a` algorithms among the `A` algorithms and apply the meta-learner. 
    Use meta-test for test score. Compute the difference between train score 
    and test score. 
    Repeat this process `repeat` times and compute the mean and std.

    Here we only use the first algorithm predicted by the meta-learner.

    N.B. For a DA matrix, we suppose the first `n_meta_train` tasks are used 
    as meta-train and the rest is used as meta-test.
    """
    n_datasets = len(da_matrix.datasets)
    if n_meta_train > n_datasets:
        raise ValueError("The number of meta-train tasks should be less than " +
                         "or equal to the total number of tasks." +
                         "But got {} > {}.".format(n_meta_train, n_datasets))
    T = n_meta_train

    A = len(da_matrix.algos)

    da_meta_train, da_meta_test = da_matrix.train_test_split(
        train_size=n_meta_train, shuffling=False
    )

    mean_tr = []
    std_tr = []
    mean_te = []
    std_te = []

    for a in range(1, A + 1):
        s_tr = []
        s_te = []
        for _ in range(repeat):
            # Choose a among A algorithms for meta-train, without replacement
            indices_algos = np.random.choice(A, a, replace=False)
            da_algo_train = da_meta_train.get_algo_subset(indices_algos)
            da_algo_test = da_meta_test.get_algo_subset(indices_algos)

            meta_learner.meta_fit(da_algo_train)
            i_algo = meta_learner.indices_algo_to_reveal[0]

            # Meta-train score
            avg_tr = np.mean(da_algo_train.perfs[:, i_algo])
            s_tr.append(avg_tr)

            # Meta-test score
            avg_te = np.mean(da_algo_test.perfs[:, i_algo])
            s_te.append(avg_te)
        
        mean_tr.append(np.mean(s_tr))
        std_tr.append(np.std(s_tr))
        mean_te.append(np.mean(s_te))
        std_te.append(np.std(s_te))

    mean_tr = np.array(mean_tr)
    std_tr = np.array(std_tr)
    mean_te = np.array(mean_te)
    std_te = np.array(std_te)

    return mean_tr, std_tr, mean_te, std_te


def plot_score_vs_n_algos_with_error_bars(repeat=100):
    datasets_dir = "../datasets"

    score_names = {
        'artificial_r50c20r20': 'Performance',
        'AutoDL': 'ALC',
        'AutoML': 'BAC or R2',
        'OpenML-Alors': 'Accuracy',
        'Statlog': 'Error rate',
    }
    
    for d in os.listdir(datasets_dir):
        # if d == 'AutoDL':
        if True:
            dataset_dir = os.path.join(datasets_dir, d)
            if os.path.isdir(dataset_dir):
                da_matrix = get_da_matrix_from_real_dataset_dir(dataset_dir)
                meta_learner = MeanMetaLearner()

                name_expe = "alc-vs-n_algos"
                if d == 'AutoDL':
                    n_meta_train = 5
                else:
                    n_meta_train = da_matrix.perfs.shape[0] // 2
                
                n_meta_test = da_matrix.perfs.shape[0] - n_meta_train

                curves = get_meta_scores_vs_n_algos(da_matrix, meta_learner, 
                    n_meta_train=n_meta_train, repeat=repeat)

                score_name = score_names[d]
                total_n_algos = len(da_matrix.algos)

                fig = plot_curve_with_error_bars(curves[0], curves[1], 
                    label='meta-train', marker='o', markersize=2)
                fig = plot_curve_with_error_bars(curves[2], curves[3], fig=fig, 
                    label='meta-test', marker='o', markersize=2)
                
                # Meta-train - meta-test
                diff_curve = curves[0] - curves[2]
                ax = fig.axes[0]
                ax.plot(np.arange(total_n_algos) + 1, diff_curve,
                    label='meta-train - meta-test', marker='o', markersize=2)

                # Theoretical bounds
                n_T = n_meta_train
                n_B = total_n_algos
                error_bars_the = [get_theoretical_error_bar(n_T, i, delta=0.05) 
                                  for i in range(1, n_B + 1)]
                ax.plot(np.arange(n_B) + 1, error_bars_the,
                    label='Theoretical error bar', marker='o', markersize=2)

                plt.xlabel("Number of algos " +
                    "(|Dtr|={}, |Dte|={}, ".format(n_meta_train, n_meta_test) +
                    "total #algos={})".format(total_n_algos))
                plt.ylabel("Average {} score".format(score_name))
                plt.legend()
                plt.title("{} - {} VS #algos".format(d, score_name))
                plt.show()
                save_fig(fig, name_expe=name_expe, 
                    filename="{}-alc-vs-n_algos.jpg".format(d))