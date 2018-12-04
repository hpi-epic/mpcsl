from flask_restful import Resource

from src.db import db
from src.master.helpers.io import load_data, marshal
from src.models import Experiment, ExperimentSchema


class ExperimentResource(Resource):
    def get(self, experiment_id):
        ds = Experiment.query.get_or_404(experiment_id)

        return marshal(ExperimentSchema, ds)


class ExperimentListResource(Resource):
    def get(self):
        ds = Experiment.query.all()

        return marshal(ExperimentSchema, ds, many=True)

    def post(self):
        data = load_data(ExperimentSchema)

        ds = Experiment(**data)

        db.session.add(ds)
        db.session.commit()

        return marshal(ExperimentSchema, ds)
