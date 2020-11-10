import sys
sys.path.insert(1, '/home/Constantin.Lange/mpci/services/executionenvironments/python/')

import numpy as np
import pandas as pd
import cProfile
import pstats

from rperm import RPermTest
from cmi_estimators import GKOVEstimator, CCMIEstimator
from py_parallel_pc import *
from py_parallel_pc import _init_worker
from py_parallel_pc import _test_worker

# Parameters
input_file = "cleaned_cmi_test.csv"
alpha = 0.05
node_1 = 0
node_2 = 1
indep_test_string = 'RPerm_GKOV'

def benchmark(level, observations, iterations):
    # Initialize
    data = pd.read_csv(input_file, sep=";", header=None, skipinitialspace=True, nrows=observations)
    data_raw = RawArray('d', data.shape[0] * data.shape[1])
    data_arr = np.frombuffer(data_raw).reshape(data.shape)
    np.copyto(data_arr, data.values)

    cols = data.columns
    vertices = len(cols)
    graph_raw = RawArray('i', np.ones(vertices*vertices).astype(int))
    graph = np.frombuffer(graph_raw, dtype="int32").reshape((vertices, vertices))

    indep_tests = {
        'RPerm_GKOV': lambda: RPermTest(GKOVEstimator(k='adaptive'), k=15,
                                        iterations=iterations),
        'RPerm_CCMI': lambda: RPermTest(CCMIEstimator(), k=15,
                                        iterations=iterations),
    }
    indep_test = indep_tests[indep_test_string]()

    _init_worker(data_raw, data.shape, graph_raw, vertices, indep_test, alpha)

    # Profile
    cProfile.run('_test_worker(node_1, node_2, '+str(level)+')', 'test_stats')
    p = pstats.Stats('test_stats')
    p.sort_stats('tottime')
    p.print_stats(5)
    print("------------------------------------------------")

benchmark(2, 10000, 10)

# Changing Permutations
#permutations = [10, 25, 50, 75, 100]
#for permutation in permutations:
#    print("Sepset: 1, Observations: 5000, Permutation: ", permutation)
#    benchmark(1, 5000, permutation)

# Changing Observations
#observations = [1000, 5000, 10000, 25000, 44999]
#for observation in observations:
#    print("Sepset: 1, Observations: "+str(observation)+", Permutation: 10")
#    benchmark(1, observation, 10)

# Changing Subsets
#levels = [1, 2, 3, 4, 5]
#for level in levels:
#   print("Sepset: "+str(level)+", Observations: 5000, Permutation: 10")
#    benchmark(level, 5000, 10)