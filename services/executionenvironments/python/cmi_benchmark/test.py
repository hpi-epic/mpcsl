import numpy as np
import pandas as pd

from rperm import RPermTest
from cmi_estimators import GKOVEstimator, CCMIEstimator
from py_parallel_pc import *

# Parameters
input_file = "cooling_data.csv"
alpha = 0.05
max_level = None
processes = 2
indep_test_string = 'RPerm_GKOV'

data = pd.read_csv(input_file, sep=";", header=None)

indep_tests = {
        'RPerm_GKOV': lambda: RPermTest(GKOVEstimator(k='adaptive'), k=15,
                                        iterations=iterations),
        'RPerm_CCMI': lambda: RPermTest(CCMIEstimator(), k=15,
                                        iterations=iterations),
    }
indep_test = indep_tests[args.independence_test]()

# Algorithm
graph, sepsets = parallel_stable_pc(df, indep_test, alpha=alpha, processes=processes, max_level=max_level)