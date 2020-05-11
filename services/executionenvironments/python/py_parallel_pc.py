# Created by Marcus Pappik
import pickle
from itertools import combinations, product
from multiprocessing import Pool, RawArray

import networkx as nx
import numpy as np

# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def _init_worker(data, data_shape, graph, vertices, test, alpha):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['data'] = data
    var_dict['data_shape'] = data_shape

    var_dict['graph'] = graph
    var_dict['vertices'] = vertices

    var_dict['alpha'] = alpha
    var_dict['test'] = test


def _test_worker(i, j, lvl):
    test = var_dict['test']
    alpha = var_dict['alpha']
    data_arr = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape'])
    graph = np.frombuffer(var_dict['graph'], dtype="int32").reshape((var_dict['vertices'],
                                                                     var_dict['vertices']))
    # unconditional
    if lvl < 1:
        p_val = test.compute_pval(data_arr[:, [i]], data_arr[:, [j]], z=None)
        if (p_val > alpha):
            return (i, j, [])
    # conditional
    else:
        candidates = np.arange(var_dict['vertices'])[(graph[i] == 1) & (graph[j] == 1)]
        if len(candidates) < lvl:
            return None

        sets = [list(S) for S in combinations(candidates, lvl)]
        for S in sets:
            if (i in S) or (j in S):
                continue
            p_val = test.compute_pval(data_arr[:, [i]], data_arr[:, [j]], z=data_arr[:, S])
            if (p_val > alpha):
                return (i, j, S)
    return None


def parallel_stable_pc(data, estimator, alpha=0.05, processes=32, return_sepsets=False, max_level=None, outfile=None):
    cols = data.columns
    cols_map = np.arange(len(cols))

    data_raw = RawArray('d', data.shape[0] * data.shape[1])
    # Wrap X as an numpy array so we can easily manipulates its data.
    data_arr = np.frombuffer(data_raw).reshape(data.shape)
    # Copy data to our shared array.
    np.copyto(data_arr, data.values)

    # same magic as for data
    vertices = len(cols)
    graph_raw = RawArray('i', np.ones(vertices*vertices).astype(int))
    graph = np.frombuffer(graph_raw, dtype="int32").reshape((vertices, vertices))
    sepsets = {} if return_sepsets else None

    lvls = range((len(cols) - 1) if max_level is None else min(len(cols)-1, max_level+1))
    for lvl in lvls:
        configs = [(i, j, lvl) for i, j in product(cols_map, cols_map) if i < j and graph[i][j] == 1]

        with Pool(processes=processes, initializer=_init_worker,
                  initargs=(data_raw, data.shape, graph_raw, vertices, estimator, alpha)) as pool:
            result = pool.starmap(_test_worker, configs)

        for r in result:
            if r is not None:
                graph[r[0]][r[1]] = 0
                graph[r[1]][r[0]] = 0
                if return_sepsets:
                    sepsets[(r[0], r[1])] = r[2]

        if outfile is not None:
            nx_graph = nx.from_numpy_matrix(graph)
            nx.relabel_nodes(nx_graph, lambda i: cols[i], copy=False)
            with open(outfile, 'wb') as file:
                pickle.dump(nx_graph, file)

    nx_graph = nx.from_numpy_matrix(graph)
    nx.relabel_nodes(nx_graph, lambda i: cols[i], copy=False)
    if outfile is not None:
        with open(outfile, 'wb') as file:
            pickle.dump({'graph': nx_graph, 'sepsets': sepsets}, file)

    return nx_graph, sepsets
