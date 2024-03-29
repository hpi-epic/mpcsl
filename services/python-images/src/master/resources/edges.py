from flask_restful import Resource
from flask_restful_swagger_2 import swagger

from src.master.helpers.io import marshal
from src.master.helpers.swagger import get_default_response
from src.models import Edge, EdgeSchema


class EdgeResource(Resource):
    @swagger.doc({
        'description': 'Returns a single edge',
        'parameters': [
            {
                'name': 'edge_id',
                'description': 'Edge identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(EdgeSchema.get_swagger()),
        'tags': ['Edge']
    })
    def get(self, edge_id):
        edge = Edge.query.get_or_404(edge_id)

        return marshal(EdgeSchema, edge)


class ResultEdgeListResource(Resource):
    @swagger.doc({
        'description': 'Returns all edges for one result',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(EdgeSchema.get_swagger().array()),
        'tags': ['Edge']
    })
    def get(self, result_id):
        edges = Edge.query.filter(Edge.result_id == result_id).all()

        return marshal(EdgeSchema, edges, many=True)


class ResultImportantEdgeListResource(Resource):

    @swagger.doc({
        'description': 'Returns all edges for one result',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'amount',
                'description': 'Amount of most significant edges to maximally return',
                'in': 'path',
                'type': 'integer',
                'required': False
            }
        ],
        'responses': get_default_response(EdgeSchema.get_swagger().array()),
        'tags': ['Edge']
    })
    def get(self, result_id, amount):
        edges = Edge.query.filter(Edge.result_id == result_id).order_by(Edge.weight.desc()).all()

        if amount and len(edges) > amount:
            edges = edges[:amount]

        return marshal(EdgeSchema, edges, many=True)
