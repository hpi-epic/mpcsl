from flask_restful import Resource
from flask_restful_swagger_2 import swagger

from src.db import db
from src.master.helpers.io import load_data, marshal
from src.master.helpers.swagger import get_default_response
from src.models import Experiment, ExperimentSchema


class ExperimentResource(Resource):
    @swagger.doc({
        'description': 'Returns a single experiment',
        'parameters': [
            {
                'name': 'experiment_id',
                'description': 'Experiment identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ExperimentSchema.get_swagger()),
        'tags': ['Experiment']
    })
    def get(self, experiment_id):
        ds = Experiment.query.get_or_404(experiment_id)

        return marshal(ExperimentSchema, ds)

    @swagger.doc({
        'description': 'Deletes a single experiment',
        'parameters': [
            {
                'name': 'experiment_id',
                'description': 'Experiment identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ExperimentSchema.get_swagger()),
        'tags': ['Experiment']
    })
    def delete(self, experiment_id):
        ds = Experiment.query.get_or_404(experiment_id)
        data = marshal(ExperimentSchema, ds)

        db.session.delete(ds)
        db.session.commit()
        return data


class ExperimentListResource(Resource):
    @swagger.doc({
        'description': 'Returns all experiments',
        'responses': get_default_response(ExperimentSchema.get_swagger().array()),
        'tags': ['Experiment']
    })
    def get(self):
        ds = Experiment.query.all()

        return marshal(ExperimentSchema, ds, many=True)

    @swagger.doc({
        'description': 'Creates an experiment',
        'parameters': [
            {
                'name': 'experiment',
                'description': 'Experiment parameters',
                'in': 'body',
                'schema': ExperimentSchema.get_swagger(True)
            }
        ],
        'responses': get_default_response(ExperimentSchema.get_swagger()),
        'tags': ['Experiment']
    })
    def post(self):
        data = load_data(ExperimentSchema)

        experiment = Experiment(**data)

        db.session.add(experiment)
        db.session.commit()

        return marshal(ExperimentSchema, experiment)
