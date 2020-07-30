# Created by Marcus Pappik
import logging
from itertools import combinations, product
from multiprocessing import Pool, RawArray
from datetime import datetime

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
            return (i, j, p_val, [])
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
                return (i, j, p_val, S)
    return None


def _unid(g, i, j):
    return g.has_edge(i, j) and not g.has_edge(j, i)


def _bid(g, i, j):
    return g.has_edge(i, j) and g.has_edge(j, i)


def _adj(g, i, j):
    return g.has_edge(i, j) or g.has_edge(j, i)


def rule1(g, j, k):
    for i in g.predecessors(j):
        # i -> j s.t. i not adjacent to k
        if _unid(g, i, j) and not _adj(g, i, k):
            g.remove_edge(k, j)
            return True
    return False


def rule2(g, i, j):
    for k in g.successors(i):
        # i -> k -> j
        if _unid(g, k, j) and _unid(g, i, k):
            g.remove_edge(j, i)
            return True
    return False


def rule3(g, i, j):
    for k, l in combinations(g.predecessors(j), 2):
        # i <-> k -> j and i <-> l -> j s.t. k not adjacent to l
        if (not _adj(g, k, l) and _bid(g, i, k) and _bid(g, i, l) and _unid(g, l, j) and _unid(g, k, j)):
            g.remove_edge(j, i)
            return True
    return False


def rule4(g, i, j):
    for l in g.predecessors(j):
        for k in g.predecessors(l):
            # i <-> k -> l -> j s.t. k not adjacent to j and i adjacent to l
            if (not _adj(g, k, j) and _adj(g, i, l) and _unid(g, k, l) and _unid(g, l, j) and _bid(g, i, k)):
                g.remove_edge(j, i)
                return True
    return False


def _direct_edges(graph, sepsets):
    digraph = nx.DiGraph(graph)
    for i in graph.nodes():
        for j in nx.non_neighbors(graph, i):
            for k in nx.common_neighbors(graph, i, j):
                sepset = sepsets[(i, j)] if (i, j) in sepsets else []
                if k not in sepset:
                    if (k, i) in digraph.edges() and (i, k) in digraph.edges():
                        digraph.remove_edge(k, i)
                    if (k, j) in digraph.edges() and (j, k) in digraph.edges():
                        digraph.remove_edge(k, j)

    bidirectional_edges = [(i, j) for i, j in digraph.edges if digraph.has_edge(j, i)]
    for i, j in bidirectional_edges:
        if _bid(digraph, i, j):
            continue
        if (rule1(digraph, i, j) or rule2(digraph, i, j) or rule3(digraph, i, j) or rule4(digraph, i, j)):
            continue

    return digraph


def parallel_stable_pc(data, estimator, alpha=0.05, processes=32, max_level=None):
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
    sepsets = {}

    lvls = range((len(cols) - 1) if max_level is None else min(len(cols)-1, max_level+1))
    for lvl in lvls:
        configs = [(i, j, lvl) for i, j in product(cols_map, cols_map) if i < j and graph[i][j] == 1]

        logging.info(f'Starting level {lvl} pool with {len(configs)} remaining edges at {datetime.now()}')
        with Pool(processes=processes, initializer=_init_worker,
                  initargs=(data_raw, data.shape, graph_raw, vertices, estimator, alpha)) as pool:
            result = pool.starmap(_test_worker, configs)

        for r in result:
            if r is not None:
                graph[r[0]][r[1]] = 0
                graph[r[1]][r[0]] = 0
                sepsets[(r[0], r[1])] = {'p_val': r[2], 'sepset': r[3]}

    nx_graph = nx.from_numpy_matrix(graph)
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    nx_digraph = _direct_edges(nx_graph, sepsets)
    nx.relabel_nodes(nx_digraph, lambda i: cols[i], copy=False)
    sepsets = {(cols[k[0]], cols[k[1]]): {'p_val': v['p_val'], 'sepset': [cols[e] for e in v['sepset']]}
               for k, v in sepsets.items()}

    return nx_digraph, sepsets
