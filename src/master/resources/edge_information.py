from flask_restful import Resource
from flask_restful_swagger_2 import swagger

from src.db import db
from src.master.helpers.io import load_data, marshal
from src.master.helpers.swagger import get_default_response
from src.models import EdgeInformation, EdgeInformationSchema


class EdgeInformationResource(Resource):

    @swagger.doc({
        'description': 'Deletes a single Edge Information',
        'parameters': [
            {
                'name': 'edge_information_id',
                'description': 'Edge Information Identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(EdgeInformationSchema.get_swagger()),
        'tags': ['Edge Information']
    })
    def delete(self, edge_information_id):
        edge_information = EdgeInformation.query.get_or_404(edge_information_id)
        data = marshal(EdgeInformationSchema, edge_information)

        db.session.delete(edge_information)
        db.session.commit()
        return data


class EdgeInformationListResource(Resource):
    @swagger.doc({
        'description': 'Returns all available Edge Information for a specific Result',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result Identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(EdgeInformationSchema.get_swagger().array()),
        'tags': ['Edge Information']
    })
    def get(self, result_id):
        edge_informations = EdgeInformation.query.filter(EdgeInformation.result_id == result_id).all()

        return marshal(EdgeInformationSchema, edge_informations, many=True)

    @swagger.doc({
        'description': 'Creates a new Edge Information',
        'parameters': [
            {
                'name': 'Edge Information',
                'description': 'Parameters for Edge Information',
                'in': 'body',
                'schema': EdgeInformationSchema.get_swagger(True)
            }
        ],
        'responses': get_default_response(EdgeInformationSchema.get_swagger()),
        'tags': ['Edge Information']
    })
    def post(self):
        data = load_data(EdgeInformationSchema)

        edge_information = EdgeInformation(**data)

        db.session.add(edge_information)
        db.session.commit()

        return marshal(EdgeInformationSchema, edge_information)
