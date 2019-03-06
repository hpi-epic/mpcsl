import numpy as np
from flask_restful_swagger_2 import swagger
from flask_restful import Resource, abort
from marshmallow import fields, validates, Schema, ValidationError

from src.db import db
from src.master.db import data_source_connections
from src.master.helpers.database import get_db_session
from src.master.helpers.io import marshal, load_data
from src.master.helpers.swagger import get_default_response, oneOf
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
        'responses': get_default_response(oneOf([DiscreteDistributionSchema,
                                                 ContinuousDistributionSchema]).get_swagger()),
        'tags': ['Node', 'Distribution']
    })
    def get(self, node_id):
        node = Node.query.get_or_404(node_id)

        dataset = node.dataset
        session = get_db_session(dataset)

        result = session.execute(f"SELECT \"{node.name}\" FROM ({dataset.load_query}) _subquery_").fetchall()
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


class DiscreteConditionSchema(Schema, SwaggerMixin):
    categorical = fields.Constant(True)
    values = fields.List(fields.String)


class ContinuousConditionSchema(Schema, SwaggerMixin):
    categorical = fields.Constant(False)
    from_value = fields.Float()
    to_value = fields.Float()


class AutoConditionSchema(Schema, SwaggerMixin):
    auto = fields.Constant(True)


class ConditionalParameterSchema(Schema, SwaggerMixin):
    conditions = fields.Dict(keys=fields.Int(), values=fields.Dict())  # Not enforced, just metadata in 2.x

    @validates('conditions')
    def validate_params(self, conds):
        for key, val in conds.items():
            if not isinstance(val.get('auto', False), bool):
                raise ValidationError(f'Field `auto` must be bool for key {key}')

            if not val.get('auto', False):
                if 'categorical' not in val or not isinstance(val['categorical'], bool):
                    raise ValidationError(f'Field `categorical` must be bool for key {key}')

                if val['categorical']:
                    if 'values' not in val or not isinstance(val['values'], list):
                        raise ValidationError(f'Field `values` must be list for key {key}')
                else:
                    if 'from_value' not in val or not (
                            isinstance(val['from_value'], int) or isinstance(val['from_value'], float)):
                        raise ValidationError(f'Field `from_value` must be numeric for key {key}')
                    if 'to_value' not in val or not (
                            isinstance(val['to_value'], int) or isinstance(val['to_value'], float)):
                        raise ValidationError(f'Field `to_value` must be numeric for key {key}')


class ConditionalContinuousDistributionSchema(ContinuousDistributionSchema):
    conditions = fields.Dict(keys=fields.Int(), values=fields.Dict())


class ConditionalDiscreteDistributionSchema(DiscreteDistributionSchema):
    conditions = fields.Dict(keys=fields.Int(), values=fields.Dict())


class ConditionalDistributionResource(Resource):

    @swagger.doc({
        'description': 'Returns the conditional distribution of an attribute as histogram values. '
                       'If the distribution is categorical, there is no bin_edges and bins '
                       'is a dictionary mapping values to counts. ',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'conditions',
                'description': 'Dictionary mapping from node id to condition. There are three types of conditions, '
                               'continuous, discrete and automatic ones where the most common value or interval '
                               'is picked. For continuous conditions, from_value and to_value represent an inclusive'
                               'interval.',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'additionalProperties': oneOf([DiscreteConditionSchema, ContinuousConditionSchema,
                                                   AutoConditionSchema]).get_swagger(True),
                    'example': {
                        '234': {
                            'categorical': True,
                            'values': ['3224', '43']
                        },
                        '4356': {
                            'categorical': False,
                            'from_value': 2.12,
                            'to_value': 2.79
                        },
                        '95652': {
                            'auto': True
                        },
                    }
                }
            }
        ],
        'responses': get_default_response(oneOf([ConditionalDiscreteDistributionSchema,
                                                 ConditionalContinuousDistributionSchema]).get_swagger()),
        'tags': ['Node', 'Distribution']
    })
    def get(self, node_id):
        node = Node.query.get_or_404(node_id)
        dataset = node.dataset
        session = get_db_session(dataset)

        conditions = load_data(ConditionalParameterSchema)['conditions']
        base_query = f"SELECT \"{node.name}\" FROM ({dataset.load_query}) _subquery_"
        predicates = []
        for condition_node_id, condition in conditions.items():
            node_name = Node.query.get_or_404(condition_node_id).name

            # Auto-generate ranges by picking largest frequency
            if condition.get('auto', False):
                node_res = session.execute(f"SELECT \"{node_name}\" "
                                           f"FROM ({dataset.load_query}) _subquery_").fetchall()
                node_data = [line[0] for line in node_res]
                if len(np.unique(node_data)) <= 10:
                    values, counts = np.unique(node_data, return_counts=True)
                    condition['values'] = [values[np.argmax(counts)]]
                    condition['categorical'] = True
                else:
                    hist, bin_edges = np.histogram(node_data, bins='auto', density=False)
                    most_common = np.argmax(hist)
                    condition['from_value'] = bin_edges[most_common]
                    condition['to_value'] = bin_edges[most_common+1]
                    condition['categorical'] = False

            if condition['categorical']:
                predicates.append(f"\"{node_name}\" IN ({','.join(map(str, condition['values']))})")
            else:
                predicates.append(f"\"{node_name}\" >= {condition['from_value']}")
                predicates.append(f"\"{node_name}\" <= {condition['to_value']}")
        query = base_query if len(predicates) == 0 else base_query + " WHERE " + ' AND '.join(predicates)

        result = session.execute(query).fetchall()
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
