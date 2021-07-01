# Author: Zhengying LIU
# Create: 4 May 2021

import numpy as np
import os
import requests

def download_file_from_google_drive(id, destination):
    """Source:
        https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def save_fig(fig, name_expe=None, results_dir='../results',
             filename=None):
    if filename is None:
        if name_expe is None:
            filename = 'learning-curves.jpg'
        else:
            filename = '{}'.format(name_expe)
            if not filename.endswith('.jpg'):
                filename += '.jpg'

    # Create directory for the experiment
    expe_dir = os.path.join(results_dir, str(name_expe))
    os.makedirs(expe_dir, exist_ok=True)
    # Save figure
    fig_path = os.path.join(expe_dir, filename)
    fig.savefig(fig_path)


def get_theoretical_error_bar(n_T, n_B, delta):
    error_bar = np.sqrt((np.log(n_B) + np.log(2 / delta)) / (2 * n_T))
    return error_bar


def get_ranking(li, negative_score=False):
    """Return the ranking of each entry in a list."""
    le = len(li)
    arr = np.array(li)
    if negative_score:
        arr = -arr
    argsort = (-arr).argsort()
    ranking = np.zeros(le)
    ranking[argsort] = np.arange(le)
    return ranking


def get_average_rank(perfs, negative_score=False):
    """
    Args:
      perfs: numpy.ndarray, performance matrix of shape (n_datasets, n_algos)
      negative_score: boolean, if True, the smaller the score is, the better
    
    Returns:
      a list of `n_algos` entries, each being the average rank of the algorithms
    
    N.B. the rank begins at 0.
    """
    if len(perfs.shape) != 2:
        raise ValueError("`perfs` should be a 2-D array.")

    n_datasets = len(perfs)
    n_algos = len(perfs[0])
    rankings = np.zeros(perfs.shape)

    rankings = perfs.argsort()
    for i, row in enumerate(perfs):
        ranking = get_ranking(row, negative_score=negative_score)
        rankings[i] = ranking
    avg_rank = rankings.mean(axis=0)
    return avg_rank