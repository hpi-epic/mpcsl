import logging
from abc import ABC, abstractmethod
from math import log

import numpy as np
from ccmi import CCMI
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from scipy.special import digamma


class Estimator(ABC):

    @abstractmethod
    def estimator_params(self):
        pass


class CMIEstimator(Estimator):

    @abstractmethod
    def compute_cmi(self, x, y, z):
        pass


class MIEstimator(CMIEstimator):

    @abstractmethod
    def compute_mi(self, x, y):
        pass

    # If estimator can do MI, CMI is just difference. Please override if needed.
    def compute_cmi(self, x, y, z):
        yz = np.concatenate([y, z], axis=1)
        mi_x_yz = self.compute_mi(x, yz)
        mi_x_z = self.compute_mi(x, z)
        logging.debug(f'{mi_x_yz} - {mi_x_z}')
        return mi_x_yz - mi_x_z


def _mixed_KSG(x, y, k=3, leafsize=16, use_python=True):
    '''
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *Mixed-KSG* mutual information estimator

        Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
        y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
        k: k-nearest neighbor parameter

        Output: one number of I(X;Y)
    '''
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N, 1))
    # dx = len(x[0])
    if y.ndim == 1:
        y = y.reshape((N, 1))
    # dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = KDTree(data, metric='chebyshev', leaf_size=leafsize) if use_python else cKDTree(data, leafsize=leafsize)
    tree_x = KDTree(x, metric='chebyshev', leaf_size=leafsize) if use_python else cKDTree(x, leafsize=leafsize)
    tree_y = KDTree(y, metric='chebyshev', leaf_size=leafsize) if use_python else cKDTree(y, leafsize=leafsize)

    def query_ball(tree, points, r):
        return (tree.query_radius(points, r, count_only=True) if use_python
                else tree.query_ball_point(points, r, p=np.inf, return_length=True))

    eps = 1e-15
    kp = np.full((N,), k)
    knn_dis = tree_xy.query(data, k+1)[0][:, k] if use_python else tree_xy.query(data, k+1, p=np.inf)[0][:, k]
    idxs = np.where(knn_dis == 0)[0]
    if idxs.size:
        kp[idxs] = query_ball(tree_xy, data[idxs], eps)

    nx = query_ball(tree_x, x, np.maximum(knn_dis - eps, eps))
    ny = query_ball(tree_y, y, np.maximum(knn_dis - eps, eps))

    ans = np.mean(digamma(kp)) - np.mean(np.log(nx)) - np.mean(np.log(ny)) + np.log(N)
    return ans


class GKOVEstimator(MIEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k = kwargs.get('k', 3)
        self.use_python = kwargs.get('use_python', True)

    def estimator_params(self):
        return {'k': self.k}

    def compute_mi(self, x, y):
        k = max(1, int(np.sqrt(len(x))/5)) if self.k == 'adaptive' else self.k
        return _mixed_KSG(x, y, k=k, use_python=self.use_python) / log(2)


class CCMIEstimator(MIEstimator):

    def estimator_params(self):
        return {}  # No configurability for now

    def compute_mi(self, x, y):
        model = CCMI(x, y, np.ones((len(x), 1)), tester='Classifier', metric='donsker_varadhan',
                     num_boot_iter=10, h_dim=64, max_ep=20)
        mi = model.get_mi_est(np.concatenate([x, y], axis=1)) / log(2)
        return mi

    def compute_cmi(self, x, y, z):
        model = CCMI(x, y, z, tester='Classifier', metric='donsker_varadhan',
                     num_boot_iter=10, h_dim=64, max_ep=20)
        cmi = model.get_cmi_est() / log(2)
        return cmi
