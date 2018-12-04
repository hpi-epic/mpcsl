import csv

import io

from flask import Response
from flask_restful import Resource

from src.db import db
from src.master.helpers.io import load_data, marshal
from src.models import Dataset, DatasetSchema


class DatasetResource(Resource):
    def get(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)

        return marshal(DatasetSchema, ds)


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
        ds = Dataset.query.get_or_404(dataset_id)

        result = db.session.execute(ds.load_query)
        keys = result.keys()

        f = io.StringIO()
        wr = csv.writer(f)
        wr.writerow(keys)
        for line in result:
            wr.writerow(line)

        resp = Response(f.getvalue(), mimetype='text/csv')
        return resp
