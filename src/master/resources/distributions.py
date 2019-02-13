import numpy as np
from flask_restful_swagger_2 import swagger
from flask_restful import Resource, abort
from marshmallow import fields, Schema

from src.db import db
from src.master.db import data_source_connections
from src.master.helpers.io import marshal, load_data
from src.master.helpers.swagger import get_default_response
from src.models import Node, BaseSchema
from src.models.swagger import SwaggerMixin


class DistributionSchema(BaseSchema, SwaggerMixin):
    node = fields.Nested('NodeSchema')
    dataset = fields.Nested('DatasetSchema')
    categorical = fields.Bool()


class ContinuousDistributionSchema(DistributionSchema):
    bins = fields.List(fields.Int())
    bin_edges = fields.List(fields.Float())
    categorical = fields.Constant(False, dump_only=True)


class DiscreteDistributionSchema(DistributionSchema):
    bins = fields.Dict(keys=fields.String(), values=fields.Int())  # Not enforced, just metadata in 2.x
    categorical = fields.Constant(True, dump_only=True)


class MarginalDistributionResource(Resource):
    @swagger.doc({
        'description': 'Returns the marginal distribution of an attribute as histogram values. '
                       'If the distribution is categorical, there is no bin_edges and bins '
                       'is a dictionary mapping values to counts',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ContinuousDistributionSchema.get_swagger()),
        'tags': ['Node', 'Distribution']
    })
    def get(self, node_id):
        node = Node.query.get_or_404(node_id)

        dataset = node.result.job.experiment.dataset
        if dataset.remote_db is not None:
            session = data_source_connections.get(dataset.remote_db, None)
            if session is None:
                abort(400)
        else:
            session = db.session

        result = session.execute(f"SELECT \"{node.name}\" FROM ({dataset.load_query}) _subquery_")
        values = [line[0] for line in result]

        if len(np.unique(values)) <= 10:  # Categorical
            bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(values, return_counts=True))])
            return marshal(DiscreteDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': bins
            })
        else:
            hist, bin_edges = np.histogram(values, bins='auto', density=False)
            return marshal(ContinuousDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': hist,
                'bin_edges': bin_edges
            })


class ConditionalParameterSchema(Schema, SwaggerMixin):
    conditions = fields.Dict(keys=fields.Int(), values=fields.Dict())  # Not enforced, just metadata in 2.x


class ConditionalContinuousDistributionSchema(ContinuousDistributionSchema):
    conditions = fields.Dict(keys=fields.Int(), values=fields.Dict())


class ConditionalDiscreteDistributionSchema(DiscreteDistributionSchema):
    conditions = fields.Dict(keys=fields.Int(), values=fields.Dict())


class ConditionalDistributionResource(Resource):

    @swagger.doc({
        'description': 'Returns the marginal distribution of an attribute as histogram values. '
                       'If the distribution is categorical, there is no bin_edges and bins '
                       'is a dictionary mapping values to counts',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ContinuousDistributionSchema.get_swagger()),
        'tags': ['Node', 'Distribution']
    })
    def get(self, node_id):
        node = Node.query.get_or_404(node_id)
        dataset = node.result.job.experiment.dataset
        if dataset.remote_db is not None:
            session = data_source_connections.get(dataset.remote_db, None)
            if session is None:
                abort(400)
        else:
            session = db.session

        conditions = load_data(ConditionalParameterSchema)['conditions']
        base_query = f"SELECT \"{node.name}\" FROM ({dataset.load_query}) _subquery_"
        predicates = []
        for condition_node_id, condition in conditions.items():
            node_name = Node.query.get_or_404(condition_node_id).name

            # Auto-generate ranges by picking largest frequency
            if condition['auto']:
                node_res = session.execute(f"SELECT \"{node_name}\" "
                                           f"FROM ({dataset.load_query}) _subquery_")
                node_data = [line[0] for line in node_res]
                if len(np.unique(node_data)) <= 10:
                    values, counts = np.unique(node_data, return_counts=True)
                    condition['values'] = [values[np.argmax(counts)]]
                    condition['categorical'] = True
                else:
                    hist, bin_edges = np.histogram(node_data, bins='auto', density=False)
                    most_common = np.argmax(hist)
                    condition['min_value'] = bin_edges[most_common]
                    condition['max_value'] = bin_edges[most_common+1]
                    condition['categorical'] = False

            if condition['categorical']:
                predicates.append(f"\"{node_name}\" IN ({','.join(map(str, condition['values']))})")
            else:
                predicates.append(f"\"{node_name}\" >= {condition['from_value']}")
                predicates.append(f"\"{node_name}\" <= {condition['to_value']}")
        query = base_query if len(predicates) == 0 else base_query + " WHERE " + ' AND '.join(predicates)

        result = session.execute(query)
        data = [line[0] for line in result]

        if len(np.unique(data)) <= 10:  # Categorical
            bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(data, return_counts=True))])
            return marshal(ConditionalDiscreteDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': bins,
                'conditions': conditions
            })
        else:
            hist, bin_edges = np.histogram(data, bins='auto', density=False)
            return marshal(ConditionalContinuousDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': hist,
                'bin_edges': bin_edges,
                'conditions': conditions
            })
