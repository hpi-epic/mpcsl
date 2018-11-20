import numpy as np
import pandas as pd
from flask import Response
from flask_restful import Resource


class Datasets(Resource):

    def get(self, dataset_id):
        mean = [0, 5, 10]
        cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
        ds = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=1000), columns=['X1', 'X2', 'X3'])
        resp = Response(ds.to_csv(index=False), mimetype='text/csv')
        return resp
