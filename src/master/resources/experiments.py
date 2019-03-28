import docker
from flask_restful import Resource
from flask_restful_swagger_2 import swagger
from marshmallow import Schema, fields
from marshmallow.validate import Length, Range, OneOf

from src.db import db
from src.master.helpers.docker import get_client
from src.master.helpers.io import load_data, marshal, InvalidInputData
from src.master.helpers.swagger import get_default_response
from src.models import Experiment, ExperimentSchema, Algorithm, Job


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

        client = get_client()
        for job in Job.query.filter(Job.experiment_id == experiment_id):
            try:
                container = client.containers.get(job.container_id)
                container.remove()
            except docker.errors.NotFound:
                pass

        db.session.delete(experiment)
        db.session.commit()
        return data


TYPE_MAP = {
    'str': lambda required, minimum, maximum, values: fields.String(required=required, validate=Length(min=1)),
    'enum': lambda required, minimum, maximum, values: fields.String(required=required, validate=OneOf(values)),
    'int': lambda required, minimum, maximum, values:
        fields.Integer(required=required, validate=Range(min=minimum, max=maximum)),
    'float': lambda required, minimum, maximum, values:
        fields.Float(required=required, validate=Range(min=minimum, max=maximum)),
    'bool': lambda required, minimum, maximum, values: fields.Boolean(required=required, validate=OneOf([True, False])),
}


def generate_schema(parameters):
    """
    This function generates a marshmallow schema,
    given a valid_parameters dict.
    The dictionary should have the structure:
    {
        *field name*: {*type*: *One of the data types listed in TYPE_MAP*,
                       *required*: *true*,  # optional
                       ...
                      }
        ...
    }
    Further examples could be found in confdefault/algorithms.json
    :param parameters: Dict of valid parameters, see above
    :return: Schema
    """
    return type('Schema', (Schema,), {
        parameter_name: TYPE_MAP[parameter_options['type']](
            parameter_options.get('required', False),
            parameter_options.get('minimum', None),
            parameter_options.get('maximum', None),
            parameter_options.get('values', None)
        )
        for parameter_name, parameter_options in parameters.items()
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
