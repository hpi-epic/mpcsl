import argparse
import logging
import time

from cmi_estimators import GKOVEstimator, CCMIEstimator
from mpci_utils import get_dataset, store_graph_result
from py_parallel_pc import parallel_stable_pc
from rperm import RPermTest

parser = argparse.ArgumentParser(description='Parse PCAlg parameters.')
parser.add_argument('-j', '--job_id', type=str, help='Job ID')
parser.add_argument('--api_host', type=str, help='API Host/Port')
parser.add_argument('-d', '--dataset_id', type=str, help='Dataset ID')
parser.add_argument('-t', '--independence_test', choices=['RPerm_GKOV', 'RPerm_CCMI'],
                    type=str, help='Independence test used for the pcalg', default='RPerm_GKOV')
parser.add_argument('-a', '--alpha', type=float, help='The significance level of the test', default=0.05)
parser.add_argument('-p', '--processes', type=int, default=1,
                    help='The number of separate processes for parallelization')
parser.add_argument('-s', '--subset_size', type=int,
                    help='The maximal size of the conditioning sets that are considered')
parser.add_argument('--permutations', type=int, default=None,
                    help='The number of iterations for the permutation test')
parser.add_argument('--send_sepsets', type=int, default=0, help='If 1, send sepsets with the results')


def run_pcalg(api_host, job_id, dataset_id, indep_test, alpha, processes, max_level, args, send_sepsets=False):
    df, ds_load_time = get_dataset(api_host, dataset_id, job_id)

    start = time.time()
    graph, sepsets = parallel_stable_pc(df, indep_test, alpha=alpha, processes=processes, max_level=max_level)
    exec_time = time.time() - start

    store_graph_result(api_host, job_id, graph, exec_time, ds_load_time, args,
                       sepsets=(sepsets if send_sepsets else None))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parser.parse_args()

    indep_tests = {
        'RPerm_GKOV': lambda: RPermTest(GKOVEstimator(k='adaptive'), k=15,
                                        iterations=args.permutations if args.permutations is not None else 100),
        'RPerm_CCMI': lambda: RPermTest(CCMIEstimator(), k=15,
                                        iterations=args.permutations if args.permutations is not None else 10),
    }
    indep_test = indep_tests[args.independence_test]()
    send_sepsets = args.send_sepsets == 1
    max_level = args.subset_size if args.subset_size >= 0 else None

    run_pcalg(args.api_host, args.job_id, args.dataset_id, indep_test, alpha=args.alpha,
              processes=args.processes, max_level=max_level, args=vars(args), send_sepsets=send_sepsets)
