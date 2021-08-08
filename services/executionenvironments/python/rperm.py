import logging
from abc import ABC, abstractmethod
from math import floor

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from tigramite import tigramite_cython_code


class IndependenceTest(ABC):

    def __init__(self, estimator):
        self.estimator = estimator

    def estimator_params(self):
        return self.test_params()

    @abstractmethod
    def test_params(self):
        pass

    @abstractmethod
    def compute_pval(self, x, y, z):
        pass


class PermutationTest(IndependenceTest):

    def __init__(self, estimator, iterations=10):
        super().__init__(estimator)
        self.iterations = iterations
        self.cmi_val = None
        self.null_distribution = None

    def compute_pval(self, x, y, z):
        if z is not None:
            raise NotImplementedError("Child class must provide conditional independence test")
        self.cmi_val = self.estimator.compute_mi(x, y)

        sig_samples = self.iterations
        sig_blocklength = max(1, len(x) // 20)
        n_blks = int(floor(float(len(x))/sig_blocklength))

        block_starts = np.arange(0, len(x) - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = x[n_blks*sig_blocklength:,]

        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):
            blk_starts = np.random.permutation(block_starts)[:n_blks]
            x_shuffled = np.zeros((n_blks*sig_blocklength, 1), dtype=x.dtype)
            for blk in range(sig_blocklength):
                x_shuffled[blk::sig_blocklength] = x[blk_starts + blk]

            # Insert tail randomly somewhere
            if tail.shape[0] > 0:
                insert_tail_at = np.random.choice(block_starts)
                x_shuffled = np.insert(x_shuffled, insert_tail_at,
                                       tail, axis=0)

            null_dist[sam] = self.estimator.compute_mi(x_shuffled, y)

        self.null_distribution = null_dist
        pval = (null_dist >= self.cmi_val).mean()
        return pval


class RPermTest(PermutationTest):

    def __init__(self, estimator, k, iterations=10, use_python=True, subsample=None):
        super().__init__(estimator, iterations)
        self.k = k
        self.use_python = use_python
        self.subsample = subsample
        self.duplicate_warnings = 1
        self.duplicate_warnings_output = 0

    def test_params(self):
        return {
            'k': self.k,
            'iterations': self.iterations,
            'estimator': self.estimator.__class__.__name__,
        }

    def compute_pval(self, x, y, z=None, recycle_cmi=False):
        if z is None:
            return super().compute_pval(x, y, z)

        if not(recycle_cmi and self.cmi_val is not None):
            self.cmi_val = self.estimator.compute_cmi(x, y, z)

        if self.subsample is not None:
            sample = np.random.choice(np.arange(len(x)), min(len(x), self.subsample), replace=False)
            x, y, z = x[sample], y[sample], z[sample]
        # Get nearest neighbors around each sample point in Z
        tree_z = KDTree(z, metric='chebyshev', leaf_size=16) if self.use_python else cKDTree(z, leafsize=16)
        neighbors = (tree_z.query(z, k=self.k+1)[1][:, 1:] if self.use_python
                     else tree_z.query(z, self.k+1, p=np.inf)[1][:, 1:]).astype('int32')

        null_dist = np.zeros(self.iterations)
        duplicate_percentage = 0
        for i in range(self.iterations):
            # Generate random order in which to go through indices loop in next step
            order = np.random.permutation(len(x)).astype('int32')
            # Select a series of neighbor indices that contains as few as possible duplicates
            restricted_permutation = tigramite_cython_code._get_restricted_permutation_cython(
                T=len(x),
                shuffle_neighbors=self.k,
                neighbors=neighbors,
                order=order
            )

            x_shuffled = x[restricted_permutation]
            duplicate_percentage = max(duplicate_percentage, 1 - len(set(restricted_permutation)) / len(x))
            null_dist[i] = self.estimator.compute_cmi(x_shuffled, y, z)
        if duplicate_percentage > 0.2:
                if self.duplicate_warnings >= pow(2,self.duplicate_warnings_output):
                    logging.warn(f'Up to {round(100*duplicate_percentage, 2)}% of permutations were duplicate, '
                                 f'consider increasing k.')
                    self.duplicate_warnings_output += 1
                self.duplicate_warnings += 1

        self.null_distribution = null_dist
        pval = (null_dist >= self.cmi_val).mean()
        return pval
