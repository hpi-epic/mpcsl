from flask_restful import Resource
from flask_restful_swagger_2 import swagger

from src.db import db
from src.master.helpers.io import load_data, marshal
from src.master.helpers.swagger import get_default_response
from src.models import Algorithm, AlgorithmSchema


class AlgorithmResource(Resource):
    @swagger.doc({
        'description': "Returns a single Algorithm",
        'parameters': [
            {
                'name': 'algorithm_id',
                'description': 'Algorithm identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(AlgorithmSchema.get_swagger()),
        'tags': ['Algorithm']
    })
    def get(self, algorithm_id):
        algorithm = Algorithm.query.get_or_404(algorithm_id)

        return marshal(AlgorithmSchema, algorithm)

    @swagger.doc({
        'description': 'Deletes a single experiment',
        'parameters': [
            {
                'name': 'algorithm_id',
                'description': 'Algorithm identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(AlgorithmSchema.get_swagger()),
        'tags': ['Algorithm']
    })
    def delete(self, algorithm_id):
        algorithm = Algorithm.query.get_or_404(algorithm_id)
        data = marshal(AlgorithmSchema, algorithm)

        db.session.delete(algorithm)
        db.session.commit()
        return data


class AlgorithmListResource(Resource):
    @swagger.doc({
        'description': 'Returns all available Algorithms',
        'responses': get_default_response(AlgorithmSchema.get_swagger().array()),
        'tags': ['Algorithm']
    })
    def get(self):
        algorithms = Algorithm.query.all()

        return marshal(AlgorithmSchema, algorithms, many=True)

    @swagger.doc({
        'description': 'Creates a new Algorithm',
        'parameters': [
            {
                'name': 'Algorithm',
                'description': 'Algorithm parameters',
                'in': 'body',
                'schema': AlgorithmSchema.get_swagger(True)
            }
        ],
        'responses': get_default_response(AlgorithmSchema.get_swagger()),
        'tags': ['Algorithm']
    })
    def post(self):
        data = load_data(AlgorithmSchema)

        algorithm = Algorithm(**data)

        db.session.add(algorithm)
        db.session.commit()

        return marshal(AlgorithmSchema, algorithm)
