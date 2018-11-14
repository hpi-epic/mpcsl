from flask_restful import Resource

from src.master.helpers.io import load_data, marshal
from src.models.dataset import DataSet, DataSetSchema


class DataSetResource(Resource):
    def get(self, data_set_id):
        ds = DataSet.get_or_404(data_set_id)

        return marshal(DataSetSchema, ds)

    def put(self, data_set_id):
        ds = DataSet.get_or_404(data_set_id)

        ds.update(load_data(DataSetSchema))

        return marshal(DataSetSchema, ds)

    def delete(self, data_set_id):
        ds = DataSet.get_or_404(data_set_id)

        self.db.session.delete(ds)

        return marshal(DataSetSchema, ds)


class DataSetListResource(Resource):
    def get(self):
        ds = DataSet.query.all()

        return marshal(DataSetSchema, ds, many=True)

    def post(self):
        data = load_data(DataSetSchema)

        ds = DataSet(**data)
        self.db.session.add(ds)

        return marshal(DataSetSchema, ds)
