# Author: Zhengying LIU
# Create: 6 May 2021

from mlt.data import DAMatrix
from mlt.data import get_da_matrix_from_real_dataset_dir
from mlt.meta_learner import MeanMetaLearner
from mlt.utils import save_fig
from mlt.utils import get_theoretical_error_bar
from mlt.utils import get_average_rank

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_curve_with_error_bars(li_mean, li_std, fig=None, label=None, xs=None, **kwargs):
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

    if xs is None:
        X = np.arange(len(li_mean)) + 1
    else:
        X = xs
    
    ax.plot(X, li_mean, label=label, **kwargs)

    ax.fill_between(X, upper, lower, alpha=0.3)
        
    return fig


def inspect_da_matrix(da_matrix, results_dir="../results", save=True):
    perfs = np.array(da_matrix.perfs)
    li_mean = np.mean(perfs, axis=0)
    li_std = np.std(perfs, axis=0)

    fig = plot_curve_with_error_bars(li_mean, li_std)
    name = da_matrix.name
    n_datasets = len(da_matrix.datasets)
    n_algos = len(da_matrix.algos)
    assert n_datasets == perfs.shape[0]
    assert n_algos == perfs.shape[1]
    title = "{} (n_datasets={}, n_algos={})".format(name, n_datasets, n_algos) 
    plt.title(title)
    if save:
        filename = "mean-std-algos-{}".format(name)
        save_fig(fig, results_dir=results_dir, filename=filename)

    heatmap = sns.clustermap(perfs, metric='correlation')
    heatmap.fig.suptitle(name)
    if save:
        heatmap.fig.savefig(os.path.join(results_dir, name))

    cov = np.corrcoef(perfs.T)
    hm_cov = sns.clustermap(cov)
    title = name + " algos correlation"
    hm_cov.fig.suptitle(title)
    if save:
        hm_cov.fig.savefig(os.path.join(results_dir, title))

    cov = np.corrcoef(perfs)
    hm_cov = sns.clustermap(cov)
    title = name + " tasks correlation"
    hm_cov.fig.suptitle(title)
    if save:
        hm_cov.fig.savefig(os.path.join(results_dir, title))

    plt.show()

    


############################
## ALC vs number of tasks ##
############################
def get_meta_scores_vs_n_tasks(da_matrix, meta_learner, 
                               n_meta_train=5,
                               repeat=100, max_ticks=50, shuffling=True):
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

    mean_tr = []
    std_tr = []
    mean_va = []
    std_va = []
    mean_te = []
    std_te = []

    step_size = max(1, T // max_ticks)
    ticks = range(1, T + 1, step_size)

    for t in ticks:
        s_tr = []
        s_va = []
        s_te = []
        for _ in range(repeat):
            # (Meta-)train-test split done in each iteration
            da_meta_train, da_meta_test = da_matrix.train_test_split(
                train_size=n_meta_train, shuffling=shuffling
            )

            # Choose t among T tasks for meta-train, without replacement
            valid_indices = set(np.random.choice(T, T - t, replace=False))
            meta_learner.meta_fit(da_meta_train, valid_indices)
            i_algo = meta_learner.indices_algo_to_reveal[0]

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

    return mean_tr, std_tr, mean_va, std_va, mean_te, std_te, ticks


def plot_score_vs_n_tasks_per_matrix(
        da_matrix, 
        meta_learner,
        repeat=100,
        log_scale=False, 
        save=False, 
        max_ticks=50,
        n_meta_train=None,
        name_expe="alc-vs-n_tasks",
        score_name="Performance",
        shuffling=False,
        **kwargs):
    """Given DA matrix `da_matrix` and meta-learn `meta_learner`, plot a score
    vs n_tasks figure. 

    Following procedures are adopted:
    - Runs are repeated experiments for computing the mean and the std in each 
      settings. The learning curves typically plot these mean and std;
    - Random meta-train-test split (matrix -> meta-train reservoir, meta-test) 
      was done once for all runs in the first version. If `shuffling`, 
      we do it in each run;
    - Random meta-train-valid split: a random subset of meta-train reservoir is 
      used for real meta-training. The remaining tasks in the meta-train 
      reservoir are used for meta-validation;
    - Gamma-level algorithm: chooses only one (beta-)algorithm during 
      meta-training. We choose the algorithm with best column mean (i.e. the 
      algorithm with the highest mean performance over tasks in meta-train) 
      among `n_algos` algorithms, which are chosen randomly.
    - Meta-test: the chosen (beta-)algorithm during meta-training is used for 
      meta-test, where the column mean of this algorithm among the meta-test 
      set is used as final score. (Thus if the meta-test set is fixed, then the 
      final scores only have a very finite set of possibilities);

    Args:
      da_matrix: `mlt.data.DAMatrix` object
      meta_learner: `mlt.meta_learner.MetaLearner` object
      repeat: int, number of repetitions for sampling meta-training examples
      log_scale: boolean. If True, x-axis and y-axis will be in log-scale
      save: boolean. If True, figures will be saved
      max_ticks: int, maximum number of ticks/points for the plot
      shuffling: boolean, whether with shuffling for (meta-)train-test split
      n_meta_train: int, number of examples used for meta-training. If `None`,
        half of the examples are used
      name_expe: str, name of the experiment. Used for naming the resulting 
        figures
      score_name: str, name of the score. Used in the figures' title
      kwargs: dict of other arguments

    Returns:
      list of curves: [mtr_mean, mtr_std, mva_mean, mva_std, mte_mean, mte_std]
    """
    if n_meta_train is None:
        n_meta_train = da_matrix.perfs.shape[0] // 2

    n_meta_test = da_matrix.perfs.shape[0] - n_meta_train

    curves = get_meta_scores_vs_n_tasks(da_matrix, meta_learner, 
        n_meta_train=n_meta_train, repeat=repeat, max_ticks=max_ticks,
        shuffling=shuffling, **kwargs)
    ticks = curves[6]

    fig = plot_curve_with_error_bars(curves[0], curves[1], xs=ticks,
        label='meta-train')
    fig = plot_curve_with_error_bars(curves[2], curves[3], fig=fig, xs=ticks,
        label='meta-valid')
    fig = plot_curve_with_error_bars(curves[4], curves[5], fig=fig, xs=ticks,
        label='meta-test')

    plt.xlabel("Number of tasks used for meta-training " +
        "(|Dtr|={}, |Dte|={})".format(n_meta_train, n_meta_test))
    plt.ylabel("Average {} score".format(score_name))
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()

    d = da_matrix.name
    plt.title("{} - {} VS #tasks".format(d, score_name))
    if save:
        save_fig(fig, name_expe=name_expe, 
            filename="{}-alc-vs-n_tasks.jpg".format(d))
    
    # Use another figure
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # Meta-train - meta-test
    diff_curve = curves[0] - curves[4]
    ax.plot(ticks, diff_curve,
        label='meta-train - meta-test', marker='o', markersize=2)

    # Theoretical bounds
    n_T = n_meta_train
    n_B = len(da_matrix.algos)
    error_bars_the = [get_theoretical_error_bar(i, n_B, delta=0.05) 
                        for i in ticks]
    ax.plot(ticks, error_bars_the,
        label='Theoretical error bar', marker='o', markersize=2)
    
    plt.xlabel("Number of tasks used for meta-training " +
        "(|Dtr|={}, |Dte|={})".format(n_meta_train, n_meta_test))
    plt.ylabel("Average {} score".format(score_name))
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()
    plt.title("{} - {} diff VS #tasks".format(d, score_name))
    plt.show()
    if save:
        save_fig(fig2, name_expe=name_expe, 
            filename="{}-alc-diff-vs-n_tasks.jpg".format(d))
    
    return curves


def plot_score_vs_n_tasks_with_error_bars(repeat=100, 
        datasets_dir="../datasets", 
        dataset_names=None, log_scale=False, save=False, max_ticks=50, 
        shuffling=False, **kwargs):
    """
    Args:
      repeat: int, number of repetitions for sampling meta-training examples
      datasets_dir: str, path to directory containing all (meta-)datasets
      dataset_names: list of str, list of dataset names to carry out the plot
      log_scale: boolean. If True, x-axis and y-axis will be in log-scale
      save: boolean. If True, figures will be saved
      max_ticks: int, maximum number of ticks/points for the plot
      shuffling: boolean, whether with shuffling for (meta-)train-test split

    Returns:
      Plots or saves several figures.
    """

    score_names = {
        'artificial_r50c20r20': 'Performance',
        'AutoDL': 'ALC',
        'AutoML': 'BAC or R2',
        'OpenML-Alors': 'Accuracy',
        'Statlog': 'Error rate',
    }

    if dataset_names is None:
        ds = os.listdir(datasets_dir)
    else:
        ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    
    for d in ds:
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
                n_meta_train=n_meta_train, repeat=repeat, max_ticks=max_ticks,
                shuffling=shuffling, **kwargs)
            ticks = curves[6]

            score_name = score_names[d] if d in score_names else 'Performance'

            fig = plot_curve_with_error_bars(curves[0], curves[1], xs=ticks,
                label='meta-train')
            fig = plot_curve_with_error_bars(curves[2], curves[3], fig=fig, xs=ticks,
                label='meta-valid')
            fig = plot_curve_with_error_bars(curves[4], curves[5], fig=fig, xs=ticks,
                label='meta-test')

            plt.xlabel("Number of tasks used for meta-training " +
                "(|Dtr|={}, |Dte|={})".format(n_meta_train, n_meta_test))
            plt.ylabel("Average {} score".format(score_name))
            if log_scale:
                plt.xscale('log')
                plt.yscale('log')
            plt.legend()
            plt.title("{} - {} VS #tasks".format(d, score_name))
            if save:
                save_fig(fig, name_expe=name_expe, 
                    filename="{}-alc-vs-n_tasks.jpg".format(d))
            
            # Use another figure
            fig2 = plt.figure()
            ax = fig2.add_subplot(1, 1, 1)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

            # Meta-train - meta-test
            diff_curve = curves[0] - curves[4]
            ax.plot(ticks, diff_curve,
                label='meta-train - meta-test', marker='o', markersize=2)

            # Theoretical bounds
            n_T = n_meta_train
            n_B = len(da_matrix.algos)
            error_bars_the = [get_theoretical_error_bar(i, n_B, delta=0.05) 
                                for i in ticks]
            ax.plot(ticks, error_bars_the,
                label='Theoretical error bar', marker='o', markersize=2)
            
            plt.xlabel("Number of tasks used for meta-training " +
                "(|Dtr|={}, |Dte|={})".format(n_meta_train, n_meta_test))
            plt.ylabel("Average {} score".format(score_name))
            if log_scale:
                plt.xscale('log')
                plt.yscale('log')
            plt.legend()
            plt.title("{} - {} diff VS #tasks".format(d, score_name))
            plt.show()
            if save:
                save_fig(fig2, name_expe=name_expe, 
                    filename="{}-alc-diff-vs-n_tasks.jpg".format(d))


#################################
## ALC vs number of algorithms ##
#################################
def get_meta_scores_vs_n_algos(da_matrix, meta_learner, 
                               n_meta_train=5,
                               repeat=100, max_ticks=50, shuffling=True, 
                               nested=False):
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

    mean_tr = []
    std_tr = []
    mean_te = []
    std_te = []

    step_size = max(1, A // max_ticks)
    ticks = range(1, A + 1, step_size)

    for idx, a in enumerate(ticks):
        s_tr = []
        s_te = []
        for _ in range(repeat):
            # (Meta-)train-test split done in each iteration
            da_meta_train, da_meta_test = da_matrix.train_test_split(
                train_size=n_meta_train, shuffling=shuffling
            )

            # Choose a among A algorithms for meta-train, without replacement
            indices_algos = np.random.choice(A, a, replace=False)
            if nested:
                indices_algos = list(range(idx + 1))
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

    return mean_tr, std_tr, mean_te, std_te, ticks


def plot_score_vs_n_algos_per_matrix(
        da_matrix, 
        meta_learner,
        repeat=100,
        log_scale=False, 
        save=False, 
        max_ticks=50,
        n_meta_train=None,
        name_expe="alc-vs-n_algos",
        score_name="Performance",
        shuffling=False,
        **kwargs):
    """Given DA matrix `da_matrix` and meta-learn `meta_learner`, plot a score
    vs n_algos figure. 

    Following procedures are adopted:
    - Runs are repeated experiments for computing the mean and the std in each 
      settings. The learning curves typically plot these mean and std;
    - Random meta-train-test split (matrix -> meta-train reservoir, meta-test) 
      was done once for all runs in the first version. If `shuffling`, 
      we do it in each run;
    - Random meta-train-valid split: a random subset of meta-train reservoir is 
      used for real meta-training. The remaining tasks in the meta-train 
      reservoir are used for meta-validation;
    - Gamma-level algorithm: chooses only one (beta-)algorithm during 
      meta-training. We choose the algorithm with best column mean (i.e. the 
      algorithm with the highest mean performance over tasks in meta-train) 
      among `n_algos` algorithms, which are chosen randomly.
    - Meta-test: the chosen (beta-)algorithm during meta-training is used for 
      meta-test, where the column mean of this algorithm among the meta-test 
      set is used as final score. (Thus if the meta-test set is fixed, then the 
      final scores only have a very finite set of possibilities);

    Args:
      da_matrix: `mlt.data.DAMatrix` object
      meta_learner: `mlt.meta_learner.MetaLearner` object
      repeat: int, number of repetitions for sampling meta-training examples
      log_scale: boolean. If True, x-axis and y-axis will be in log-scale
      save: boolean. If True, figures will be saved
      max_ticks: int, maximum number of ticks/points for the plot
      shuffling: boolean, whether with shuffling for (meta-)train-test split
      n_meta_train: int, number of examples used for meta-training. If `None`,
        half of the examples are used
      name_expe: str, name of the experiment. Used for naming the resulting 
        figures
      score_name: str, name of the score. Used in the figures' title
      kwargs: dict of other arguments, which is passed to the function
        `get_meta_scores_vs_n_algos`.

    Returns:
      list of curves: [mtr_mean, mtr_std, mte_mean, mte_std]
    """
    if n_meta_train is None:
        n_meta_train = da_matrix.perfs.shape[0] // 2

    n_meta_test = da_matrix.perfs.shape[0] - n_meta_train

    curves = get_meta_scores_vs_n_algos(da_matrix, meta_learner, 
        n_meta_train=n_meta_train, repeat=repeat, 
        max_ticks=max_ticks,
        shuffling=shuffling,
        **kwargs)
    ticks = curves[4]

    total_n_algos = len(da_matrix.algos)

    fig = plot_curve_with_error_bars(curves[0], curves[1], xs=ticks,
        label='meta-train', marker='o', markersize=2)
    fig = plot_curve_with_error_bars(curves[2], curves[3], fig=fig, xs=ticks,
        label='meta-test', marker='o', markersize=2)
    plt.legend()
    plt.xlabel("Number of algos " +
        "(|Dtr|={}, |Dte|={}, ".format(n_meta_train, n_meta_test) +
        "total #algos={})".format(total_n_algos))
    plt.ylabel("Average {} score".format(score_name))
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    # Use another figure
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # Meta-train - meta-test
    diff_curve = curves[0] - curves[2]
    ax.plot(ticks, diff_curve,
        label='meta-train - meta-test', marker='o', markersize=2)

    # Theoretical bounds
    n_T = n_meta_train
    n_B = total_n_algos
    error_bars_the = [get_theoretical_error_bar(n_T, i, delta=0.05) 
                        for i in ticks]
    ax.plot(ticks, error_bars_the,
        label='Theoretical error bar', marker='o', markersize=2)

    # Figure's
    plt.xlabel("Number of algos " +
        "(|Dtr|={}, |Dte|={}, ".format(n_meta_train, n_meta_test) +
        "total #algos={})".format(total_n_algos))
    plt.ylabel("Average {} score".format(score_name))
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    # Title
    d = da_matrix.name
    fig.axes[0].set_title("{} - {} VS #algos".format(d, score_name))
    fig2.axes[0].set_title("{} - {} diff VS #algos".format(d, score_name))
    plt.legend()
    plt.show()
    if save:
        save_fig(fig, name_expe=name_expe, 
            filename="{}-alc-vs-n_algos.jpg".format(d))
        save_fig(fig2, name_expe=name_expe, 
            filename="{}-alc-diff-vs-n_algos.jpg".format(d))

    return curves


def plot_score_vs_n_algos_with_error_bars(repeat=100,
        datasets_dir="../datasets", 
        dataset_names=None, log_scale=False, save=False, max_ticks=50,
        shuffling=False, **kwargs):
    """
    Args:
      repeat: int, number of repetitions for sampling meta-training examples
      datasets_dir: str, path to directory containing all (meta-)datasets
      dataset_names: list of str, list of dataset names to carry out the plot
      log_scale: boolean. If True, x-axis and y-axis will be in log-scale
      save: boolean. If True, figures will be saved
      max_ticks: int, maximum number of ticks/points for the plot
      shuffling: boolean, whether with shuffling for (meta-)train-test split

    Returns:
      Plots or saves several figures.
    """

    score_names = {
        'artificial_r50c20r20': 'Performance',
        'AutoDL': 'ALC',
        'AutoML': 'BAC or R2',
        'OpenML-Alors': 'Accuracy',
        'Statlog': 'Error rate',
    }
    
    if dataset_names is None:
        ds = os.listdir(datasets_dir)
    else:
        ds = [d for d in os.listdir(datasets_dir) if d in set(dataset_names)]
    
    for d in ds:
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
                    n_meta_train=n_meta_train, repeat=repeat, 
                    max_ticks=max_ticks,
                    shuffling=shuffling,
                    **kwargs)
                ticks = curves[4]

                score_name = score_names[d] if d in score_names else 'Performance'
                total_n_algos = len(da_matrix.algos)

                fig = plot_curve_with_error_bars(curves[0], curves[1], xs=ticks,
                    label='meta-train', marker='o', markersize=2)
                fig = plot_curve_with_error_bars(curves[2], curves[3], fig=fig, xs=ticks,
                    label='meta-test', marker='o', markersize=2)
                plt.legend()
                plt.xlabel("Number of algos " +
                    "(|Dtr|={}, |Dte|={}, ".format(n_meta_train, n_meta_test) +
                    "total #algos={})".format(total_n_algos))
                plt.ylabel("Average {} score".format(score_name))
                if log_scale:
                    plt.xscale('log')
                    plt.yscale('log')

                # Use another figure
                fig2 = plt.figure()
                ax = fig2.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

                # Meta-train - meta-test
                diff_curve = curves[0] - curves[2]
                ax.plot(ticks, diff_curve,
                    label='meta-train - meta-test', marker='o', markersize=2)

                # Theoretical bounds
                n_T = n_meta_train
                n_B = total_n_algos
                error_bars_the = [get_theoretical_error_bar(n_T, i, delta=0.05) 
                                  for i in ticks]
                ax.plot(ticks, error_bars_the,
                    label='Theoretical error bar', marker='o', markersize=2)

                if d == 'OpenML-Alors':
                    plt.xscale('log')

                plt.xlabel("Number of algos " +
                    "(|Dtr|={}, |Dte|={}, ".format(n_meta_train, n_meta_test) +
                    "total #algos={})".format(total_n_algos))
                plt.ylabel("Average {} score".format(score_name))
                if log_scale:
                    plt.xscale('log')
                    plt.yscale('log')
                # Title
                fig.axes[0].set_title("{} - {} VS #algos".format(d, score_name))
                fig2.axes[0].set_title("{} - {} diff VS #algos".format(d, score_name))
                plt.legend()
                plt.show()
                if save:
                    save_fig(fig, name_expe=name_expe, 
                        filename="{}-alc-vs-n_algos.jpg".format(d))
                    save_fig(fig2, name_expe=name_expe, 
                        filename="{}-alc-diff-vs-n_algos.jpg".format(d))


def plot_all_figures(repeat=100, datasets_dir="../datasets", 
        dataset_names=None, log_scale=False, save=True, max_ticks=50, 
        shuffling=False, **kwargs):
    plot_score_vs_n_algos_with_error_bars(repeat=repeat,
        datasets_dir=datasets_dir, 
        dataset_names=dataset_names, log_scale=log_scale, 
        save=save, max_ticks=max_ticks, shuffling=shuffling, **kwargs)
    plot_score_vs_n_tasks_with_error_bars(repeat=repeat,
        datasets_dir=datasets_dir, 
        dataset_names=dataset_names, log_scale=log_scale,
        save=save, max_ticks=max_ticks, shuffling=shuffling, **kwargs)


#####################################
### Top K Meta-Learner Comparison ###
#####################################
def get_meta_learner_avg_rank(da_tr, da_te, meta_learner, repeat=10):
    n_algos = len(da_tr.algos)
    perfs_te = da_te.perfs
    avg_ranks_te = get_average_rank(perfs_te)

    avg_ranks_fit = []
    ks = []
    for i in range(repeat):
        meta_learner.meta_fit(da_tr)
        try:
            print(meta_learner.name, da_tr.name, len(da_tr.algos), meta_learner.k)
            ks.append(meta_learner.k)
        except:
            print("No info on k.")
        idx = meta_learner.indices_algo_to_reveal[0]
        print("Chosen algorithm: {}".format(da_tr.algos[idx].name))
        ar = avg_ranks_te[idx]
        avg_ranks_fit.append(ar)
    
    mean = np.mean(avg_ranks_fit)
    std = np.std(avg_ranks_fit)

    return mean, std, ks


def plot_meta_learner_comparison(da_tr, da_te, meta_learners, repeat=10, 
        save=True):
    n_algos = len(da_tr.algos)
    
    means = []
    stds = []
    kss = []
    for i, meta_learner in enumerate(meta_learners):
        mean, std, ks = get_meta_learner_avg_rank(
            da_tr, da_te, meta_learner, repeat=10)
        means.append(mean)
        stds.append(std)
        kss.append(ks)
    
    x_pos = np.arange(len(meta_learners))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Average rank in percentage')
    ax.set_xticks(x_pos)
    names = [meta_learner.name for meta_learner in meta_learners]
    ax.set_xticklabels(names)
    for i in range(len(meta_learners)):
        ks = kss[i]
        kmean = np.mean(ks)
        kstd = np.std(ks)
        if kstd == 0:
            s = "k={}".format(kmean)
        else:
            s = "k={:.1f}±{:.1f}".format(kmean, kstd)
        x = x_pos[i] - 0.2
        y = means[i] * 0.9 - 1
        plt.text(x, y, s)
    da_name = da_tr.name[:-11]
    title = "Meta-learner comparison on {} (n_algos={})".format(da_name, n_algos)
    ax.set_title(title)

    # Save the figure and show
    plt.tight_layout()
    plt.show()

    name_expe = 'meta-learner-comparison'
    filename = '{}.png'.format(da_name.lower())

    if save:
        save_fig(fig, name_expe=name_expe, filename=filename)
        
