from flask_restful import Resource
from flask_restful_swagger_2 import swagger
from marshmallow import Schema, fields
from marshmallow.validate import Length

from src.db import db
from src.master.helpers.io import load_data, marshal, InvalidInputData
from src.master.helpers.swagger import get_default_response
from src.models import Experiment, ExperimentSchema, Algorithm


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
        experiment = Experiment.query.get_or_404(experiment_id)

        return marshal(ExperimentSchema, experiment)

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
        experiment = Experiment.query.get_or_404(experiment_id)
        data = marshal(ExperimentSchema, experiment)

        db.session.delete(experiment)
        db.session.commit()
        return data


TYPE_MAP = {
    'str': lambda: fields.String(required=True, validate=Length(min=0)),
    'int': lambda: fields.Integer(required=True),
    'float': lambda: fields.Float(required=True),
    'bool': lambda: fields.Boolean(required=True),
}


def generate_schema(fields):
    """
    This function generates a marshmallow schema,
    given a fields dicts.
    The dictionary should have the structure:
    {
        *field name*: *One of the datatypes listed in TYPE_MAP*,
        ...
    }
    :param fields: List of fields, see above
    :return: Schema
    """
    return type('Schema', (Schema,), {
        attribute: TYPE_MAP[fields[attribute]]() for attribute in fields.keys()
    })


class ExperimentListResource(Resource):
    @swagger.doc({
        'description': 'Returns all experiments',
        'responses': get_default_response(ExperimentSchema.get_swagger().array()),
        'tags': ['Experiment']
    })
    def get(self):
        experiments = Experiment.query.all()

        return marshal(ExperimentSchema, experiments, many=True)

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

        algorithm = Algorithm.query.get_or_404(data['algorithm_id'])
        schema = generate_schema(algorithm.valid_parameters)
        params, errors = schema().load(data['parameters'])

        if len(errors) > 0:
            raise InvalidInputData(payload=errors)

        data['parameters'] = params

        experiment = Experiment(**data)

        db.session.add(experiment)
        db.session.commit()

        return marshal(ExperimentSchema, experiment)
