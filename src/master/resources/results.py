from flask import Response
from flask_restful import Resource, reqparse
from flask_restful_swagger_2 import swagger
from marshmallow import fields
import networkx as nx
from werkzeug.exceptions import BadRequest

from src.db import db
from src.master.helpers.io import marshal
from src.master.helpers.swagger import get_default_response
from src.models import Result, ResultSchema, Node, NodeSchema, Edge
from src.models.swagger import SwaggerMixin


class ResultListResource(Resource):

    @swagger.doc({
        'description': 'Returns all available results',
        'responses': get_default_response(ResultSchema.get_swagger().array()),
        'tags': ['Result']
    })
    def get(self):
        results = Result.query.all()

        return marshal(ResultSchema, results, many=True)


class ResultLoadSchema(ResultSchema, SwaggerMixin):
    nodes = fields.Nested('NodeSchema', many=True)
    edges = fields.Nested('EdgeSchema', many=True)
    sepsets = fields.Nested('SepsetSchema', many=True)


class ResultResource(Resource):
    @swagger.doc({
        'description': 'Returns a single result including nodes and edges',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ResultLoadSchema.get_swagger()),
        'tags': ['Result']
    })
    def get(self, result_id):
        result = Result.query.get_or_404(result_id)
        result_json = marshal(ResultLoadSchema, result)
        nodes = Node.query.filter_by(dataset_id=result.job.experiment.dataset_id).all()
        result_json['nodes'] = marshal(NodeSchema, nodes, many=True)

        return result_json

    @swagger.doc({
        'description': 'Deletes a single result',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ResultSchema.get_swagger()),
        'tags': ['Result']
    })
    def delete(self, result_id):
        result = Result.query.get_or_404(result_id)
        data = marshal(ResultSchema, result)

        db.session.delete(result)
        db.session.commit()
        return data


class GraphExportResource(Resource):
    @swagger.doc({
        'description': 'Returns the complete graph in a GraphML',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'format',
                'description': 'Graph export format',
                'in': 'query',
                'type': 'string',
                'enum': ['GEXF', 'GraphML', 'GML'],
                'default': 'GEXF'
            }
        ],
        'responses': get_default_response(ResultLoadSchema.get_swagger()),
        'tags': ['Result']
    })
    def get(self, result_id):
        result = Result.query.get_or_404(result_id)

        parser = reqparse.RequestParser()
        parser.add_argument('format', required=False, type=str, store_missing=False)
        args = parser.parse_args()
        format_type = args.get('format', 'gexf').lower()
        supported_types = ['gexf', 'graphml', 'gml']
        if format_type not in supported_types:
            raise BadRequest(f'Graph format `{format_type}` is not supported. Supported types are: {supported_types}')

        nodes = Node.query.filter_by(dataset_id=result.job.experiment.dataset_id).all()
        edges = Edge.query.filter_by(result_id=result_id).all()

        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(node.id, name=node.name)
        for edge in edges:
            graph.add_edge(edge.from_node, edge.to_node, weight=edge.weight)

        if format_type == 'gexf':
            return Response(nx.generate_gexf(graph), mimetype='text/xml')
        elif format_type == 'graphml':
            return Response(nx.generate_graphml(graph), mimetype='text/xml')
        elif format_type == 'gml':
            return Response(nx.generate_gml(graph, nx.readwrite.gml.literal_stringizer), mimetype='text/plain')
