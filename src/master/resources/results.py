from flask import Response
from flask_restful import Resource, reqparse
from flask_restful_swagger_2 import swagger
from marshmallow import fields
from werkzeug.exceptions import BadRequest
import json
import networkx as nx

from src.db import db
from src.master.helpers.io import marshal
from src.master.helpers.swagger import get_default_response
from src.models import Result, ResultSchema, Node, NodeSchema, EdgeInformation
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
    supported_types = ['GEXF', 'GraphML', 'GML', 'node_link_data.json']

    @swagger.doc({
        'description': 'Returns the complete graph in a graph file format',
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
                'enum': supported_types,
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
        if format_type not in [x.lower() for x in self.supported_types]:
            raise BadRequest(f'Graph format `{format_type}` is not one of the supported types: {self.supported_types}')

        graph = nx.DiGraph(id=str(result_id), name=f'Graph_{result_id}')
        for node in result.job.experiment.dataset.nodes:
            graph.add_node(node.id, label=node.name)
        for edge in result.edges:
            edge_info = EdgeInformation.query.filter_by(edge=edge).one_or_none()
            edge_label = edge_info.annotation.name if edge_info else ''
            graph.add_edge(edge.from_node.id, edge.to_node.id, id=edge.id, label=edge_label, weight=edge.weight)

        headers = {'Content-Disposition': f'attachment;filename=Graph_{result_id}.{format_type}'}
        if format_type == 'gexf':
            return Response(nx.generate_gexf(graph), mimetype='text/xml', headers=headers)
        elif format_type == 'graphml':
            return Response(nx.generate_graphml(graph), mimetype='text/xml', headers=headers)
        elif format_type == 'gml':
            return Response(nx.generate_gml(graph), mimetype='text/plain', headers=headers)
        elif format_type == 'node_link_data.json':
            return Response(json.dumps(nx.readwrite.json_graph.node_link_data(graph)),
                            mimetype='application/json', headers=headers)
