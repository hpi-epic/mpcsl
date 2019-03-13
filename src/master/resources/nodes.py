from flask_restful_swagger_2 import swagger
from flask_restful import Resource
from marshmallow import fields

from src.master.helpers.database import load_networkx_graph
from src.master.helpers.graph import get_potential_confounders
from src.master.helpers.io import marshal
from src.master.helpers.swagger import get_default_response
from src.models import Node, NodeSchema, BaseSchema, Result
from src.models.swagger import SwaggerMixin


class NodeResource(Resource):
    @swagger.doc({
        'description': 'Returns a single node',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(NodeSchema.get_swagger()),
        'tags': ['Node']
    })
    def get(self, node_id):
        node = Node.query.get_or_404(node_id)

        return marshal(NodeSchema, node)


class ResultNodeListResource(Resource):
    @swagger.doc({
        'description': 'Returns all nodes for one result',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(NodeSchema.get_swagger().array()),
        'tags': ['Node']
    })
    def get(self, result_id):
        result = Result.query.get_or_404(result_id)
        nodes = Node.query.filter(Node.dataset_id == result.job.experiment.dataset_id).all()

        return marshal(NodeSchema, nodes, many=True)


class NodeContextSchema(BaseSchema, SwaggerMixin):
    main_node = fields.Nested('NodeSchema')
    context_nodes = fields.Nested('NodeSchema', many=True)
    edges = fields.Nested('EdgeSchema', many=True)


class NodeContextResource(Resource):
    @swagger.doc({
        'description': 'Returns all nodes for one result',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(NodeContextSchema.get_swagger()),
        'tags': ['Node']
    })
    def get(self, node_id):
        main_node = Node.query.get_or_404(node_id)
        edges = main_node.edge_froms + main_node.edge_tos
        context_nodes = {n for edge in edges for n in [edge.from_node, edge.to_node] if n != main_node}

        return marshal(NodeContextSchema, {
            'main_node': main_node,
            'context_nodes': list(context_nodes),
            'edges': edges
        })


class NodeConfounderSchema(BaseSchema, SwaggerMixin):
    confounders = fields.List(fields.List(fields.Int))


class NodeConfounderResource(Resource):

    @swagger.doc({
        'description': 'Returns confounders to condition for an intervention. '
                       'If any neighbors are bidirectional, this will return multiple lists of node IDs '
                       'as the equivalence class contains many possible DAGs.',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(NodeContextSchema.get_swagger()),
        'tags': ['Node']
    })
    def get(self, node_id, result_id):
        node = Node.query.get_or_404(node_id)
        result = Result.query.get_or_404(result_id)

        graph = load_networkx_graph(result)
        confounder_list = get_potential_confounders(graph, node.id)

        return marshal(NodeConfounderSchema, {
            'confounders': confounder_list
        })
