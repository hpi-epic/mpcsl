import json
import logging
import os
import pickle
import sys
import time
from io import StringIO

import pandas as pd
import requests


def handle_request(request_func, api_host, job_id):
    s = requests.Session()
    s.mount('http://', requests.adapters.HTTPAdapter(max_retries=5))
    s.mount('https://', requests.adapters.HTTPAdapter(max_retries=5))
    r = request_func(s)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        if not os.path.exists('/logs'):
            os.makedirs('/logs')
        pickle.dump(r, open(f'/logs/job_{job_id}_error.pickle', 'wb'))
        s.put(f'http://{api_host}/api/job/{job_id}')
        raise
    return r


def get_dataset(api_host, dataset_id, job_id, sampling_method='random', sampling_factor=1.0):
    url = f'http://{api_host}/api/dataset/{dataset_id}/loadwithids'
    logging.info(f'Load dataset from {url}')
    start = time.time()
    r = handle_request(lambda s: s.get(url, timeout=None), api_host, job_id)
    dataset_loading_time = time.time() - start
    logging.info(f"Successfully loaded dataset (size {r.headers['x-content-length']} bytes) in {dataset_loading_time}")
    df = pd.read_csv(StringIO(r.text))
    if sampling_method == 'random':
        df = df.sample(frac=sampling_factor)
    else:
        df = df.head(int(len(df) * sampling_factor))
    return df, dataset_loading_time

def estimate_weight():
    #TODO implement
    return 0

def store_graph_result(api_host, job_id, graph, exec_time, ds_load_time, args, sepsets=None):
    edge_list = []
    for a, b in graph.edges():
        edge_list.append({'from_node': int(a), 'to_node': int(b), 'weight': estimate_weight()})

    sepset_list = []
    if sepsets is not None:
        for (a, b), sepset in sepsets.items():
            sepset_list.append({
                'from_node': int(a),
                'to_node': int(b),
                'statistic': sepset['p_val'],
                'level': len(sepset['sepset']),
                'nodes': sepset['sepset'],
            })

    payload = {
        'job_id': job_id,
        'edge_list': edge_list,
        'meta_results': args,
        'sepset_list': sepset_list,
        'execution_time': exec_time,
        'dataset_loading_time': ds_load_time,
    }
    logging.info(f'Storing graph result... (size {sys.getsizeof(payload)})')
    r = handle_request(lambda s: s.post(f'http://{api_host}/api/job/{job_id}/result', data=json.dumps(payload)),
                       api_host, job_id)
    logging.info(f'Stored graph successfully')
    return r
