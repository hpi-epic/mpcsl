import numpy as np
import pandas as pd
from flask import Response
from flask_restful import Resource

from src.db import db
from src.master.helpers.io import load_data, marshal
from src.models import Dataset, DatasetSchema


class DatasetResource(Resource):
    def get(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)

        return marshal(DatasetSchema, ds)

    def put(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)

        ds.update(load_data(DatasetSchema))

        db.session.commit()

        return marshal(DatasetSchema, ds)

    def delete(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)

        data = marshal(DatasetSchema, ds)
        db.session.delete(ds)

        db.session.commit()

        return data


class DatasetListResource(Resource):
    def get(self):
        ds = Dataset.query.all()

        return marshal(DatasetSchema, ds, many=True)

    def post(self):
        data = load_data(DatasetSchema)

        ds = Dataset(**data)

        db.session.add(ds)
        db.session.commit()

        return marshal(DatasetSchema, ds)


class DatasetLoadResource(Resource):

    def get(self, dataset_id):
        mean = [0, 5, 10]
        cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
        ds = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=1000), columns=['X1', 'X2', 'X3'])
        resp = Response(ds.to_csv(index=False), mimetype='text/csv')
        return resp
