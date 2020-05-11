import json
import logging
import pickle
import time
from io import StringIO

import pandas as pd
import requests


def handle_request(request_func, api_host, job_id):
    r = request_func()
    try:
        r.raise_for_status()
    except requests.HTTPError:
        pickle.dump(r, open(f'/logs/job_{job_id}_error.pickle'))
        requests.put(f'http://{api_host}/api/job/{job_id}')
        raise
    return r


def get_dataset(api_host, dataset_id, job_id):
    url = f'http://{api_host}/api/dataset/{dataset_id}/loadwithids'
    logging.info(f'Load dataset from {url}')
    start = time.time()
    r = handle_request(lambda: requests.get(url), api_host, job_id)
    dataset_loading_time = time.time() - start
    logging.info(f"Successfully loaded dataset (size {r.headers['x-content-length']} bytes) in {dataset_loading_time}")
    df = pd.read_csv(StringIO(r.text))
    return df, dataset_loading_time


def store_graph_result(api_host, job_id, graph, exec_time, ds_load_time, sepsets=None):
    sepset_list = []
    if sepsets is not None:
        pass

    payload = {
        'job_id': job_id,
        'edge_list': [],
        'meta_results': {},
        'sepset_list': [],
        'execution_time': exec_time,
        'dataset_loading_time': ds_load_time,
    }
    r = handle_request(lambda: requests.post(f'http://{api_host}/api/job/{job_id}/result', data=json.dumps(payload)),
                       api_host, job_id)
    return r