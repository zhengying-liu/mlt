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
            filename = '{}.jpg'.format(name_expe)

    # Create directory for the experiment
    expe_dir = os.path.join(results_dir, str(name_expe))
    os.makedirs(expe_dir, exist_ok=True)
    # Save figure
    fig_path = os.path.join(expe_dir, filename)
    fig.savefig(fig_path)


def get_theoretical_error_bar(n_T, n_B, delta):
    error_bar = np.sqrt((np.log(n_B) + np.log(2 / delta)) / (2 * n_T))
    return error_bar