import numpy as np
import pandas as pd
from flask_restful import Resource
from flask_restful_swagger_2 import swagger
from marshmallow import fields, validates, Schema, ValidationError

from src.master.helpers.database import get_db_session
from src.master.helpers.io import marshal, load_data, InvalidInputData
from src.master.helpers.swagger import get_default_response, oneOf
from src.models import Node, BaseSchema
from src.models.swagger import SwaggerMixin

DISCRETE_LIMIT = 10


def _custom_histogram(arr, max_bins=20, **kwargs):
    # Use 'auto' binning, but only up to 20 bins
    arr = np.asarray(arr)
    first_edge, last_edge = arr.min(), arr.max()
    width = np.lib.histograms._hist_bin_auto(arr, (first_edge, last_edge))
    bin_count = min(max_bins, int(np.ceil((last_edge - first_edge) / width))) if width else 1
    return np.histogram(arr, bins=bin_count, **kwargs)


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

        if len(np.unique(values)) <= DISCRETE_LIMIT:  # Categorical
            bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(values, return_counts=True))])
            return marshal(DiscreteDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': bins
            })
        else:
            hist, bin_edges = _custom_histogram(values, density=False)
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
    def post(self, node_id):
        node = Node.query.get_or_404(node_id)
        dataset = node.dataset
        session = get_db_session(dataset)

        conditions = load_data(ConditionalParameterSchema)['conditions']
        base_query = f"SELECT \"{node.name}\" FROM ({dataset.load_query}) _subquery_"

        base_result = session.execute(base_query).fetchall()
        _, node_bins = _custom_histogram([line[0] for line in base_result])

        predicates = []
        for condition_node_id, condition in conditions.items():
            node_name = Node.query.get_or_404(condition_node_id).name

            # Auto-generate ranges by picking largest frequency
            if condition.get('auto', False):
                node_res = session.execute(f"SELECT \"{node_name}\" "
                                           f"FROM ({dataset.load_query}) _subquery_").fetchall()
                node_data = [line[0] for line in node_res]
                if len(np.unique(node_data)) <= DISCRETE_LIMIT:
                    values, counts = np.unique(node_data, return_counts=True)
                    condition['values'] = [int(values[np.argmax(counts)])]
                    condition['categorical'] = True
                else:
                    hist, bin_edges = _custom_histogram(node_data, density=False)
                    most_common = np.argmax(hist)
                    condition['from_value'] = bin_edges[most_common]
                    condition['to_value'] = bin_edges[most_common+1]
                    condition['categorical'] = False

            if condition['categorical']:
                predicates.append(f"\"{node_name}\" IN ({','.join(map(repr, condition['values']))})")
            else:
                predicates.append(f"\"{node_name}\" >= {repr(condition['from_value'])}")
                predicates.append(f"\"{node_name}\" <= {repr(condition['to_value'])}")
        categorical_check = session.execute(f"SELECT 1 FROM ({dataset.load_query}) _subquery_ "
                                            f"HAVING COUNT(DISTINCT \"{node.name}\") <= {DISCRETE_LIMIT}").fetchall()
        is_categorical = len(categorical_check) > 0

        query = base_query if len(predicates) == 0 else base_query + " WHERE " + ' AND '.join(predicates)
        result = session.execute(query).fetchall()
        data = [line[0] for line in result]

        if is_categorical:  # Categorical
            bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(data, return_counts=True))])
            return marshal(ConditionalDiscreteDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': bins,
                'conditions': conditions
            })
        else:
            hist, bin_edges = np.histogram(data, bins=node_bins, density=False)
            return marshal(ConditionalContinuousDistributionSchema, {
                'node': node,
                'dataset': dataset,
                'bins': hist,
                'bin_edges': bin_edges,
                'conditions': conditions
            })


class InterventionalParameterSchema(Schema, SwaggerMixin):
    cause_node_id = fields.Int(required=True)
    effect_node_id = fields.Int(required=True)
    factor_node_ids = fields.List(fields.Int(), default=[])
    cause_condition = fields.Dict()

    @validates('cause_condition')
    def validate_params(self, cond):
        if DiscreteConditionSchema().validate(cond) and ContinuousConditionSchema().validate(cond):  # errors on both
            raise ValidationError(f'Condition must conform to DiscreteConditionSchema or ContinuousConditionSchema')


class InterventionalDistributionResource(Resource):

    @swagger.doc({
        'description': '',
        'parameters': [
            {
                'name': 'cause_node_id',
                'description': 'Node identifier of cause',
                'in': 'body',
                'schema': {
                    'type': 'integer',
                },
                'required': True
            },
            {
                'name': 'effect_node_id',
                'description': 'Node identifier of effect',
                'in': 'body',
                'schema': {
                    'type': 'integer',
                },
                'required': True
            },
            {
                'name': 'factor_node_ids',
                'description': 'Node identifiers of external factors',
                'in': 'body',
                'schema': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                },
                'default': []
            },
            {
                'name': 'cause_condition',
                'description': 'Interventional value(s) of cause',
                'required': True,
                'in': 'body',
                'schema': oneOf([DiscreteConditionSchema, ContinuousConditionSchema]).get_swagger(True)
            }
        ],
        'responses': get_default_response(oneOf([DiscreteDistributionSchema,
                                                 ContinuousDistributionSchema]).get_swagger()),
        'tags': ['Node', 'Distribution']
    })
    def post(self):
        data = load_data(InterventionalParameterSchema)
        cause_condition = data['cause_condition']

        cause_node = Node.query.get_or_404(data['cause_node_id'])
        effect_node = Node.query.get_or_404(data['effect_node_id'])
        try:
            factor_node_ids = data.get('factor_node_ids', [])
            factor_nodes = [Node.query.get_or_404(factor_node_id) for factor_node_id in factor_node_ids]
        except ValueError:
            raise InvalidInputData('factor_node_ids must be array of ints')

        if effect_node in factor_nodes:
            raise InvalidInputData('The effect cannot be a predecessor of the cause')
        dataset = effect_node.dataset
        if not all([n.dataset == dataset for n in [cause_node] + factor_nodes]):
            raise InvalidInputData('Nodes are not all from same dataset')
        session = get_db_session(dataset)

        categorical_query = (f"SELECT 1 FROM ({dataset.load_query}) _subquery_ HAVING "
                             f"COUNT(DISTINCT \"{effect_node.name}\") <= {DISCRETE_LIMIT} AND "
                             f"COUNT(DISTINCT \"{cause_node.name}\") <= {DISCRETE_LIMIT}")
        for factor_node in factor_nodes:
            categorical_query += f" AND COUNT(DISTINCT \"{factor_node.name}\") <= {DISCRETE_LIMIT}"
        categorical_check = session.execute(categorical_query).fetchall()
        is_fully_categorical = len(categorical_check) > 0

        if is_fully_categorical:  # Categorical, can be done in DB
            # cause c, effect e, factors F
            # P(e|do(c)) = \Sigma_{F} P(e|c,f) P(f)
            category_query = session.execute(f"SELECT DISTINCT \"{effect_node.name}\" "
                                             f"FROM ({dataset.load_query}) _subquery_").fetchall()
            categories = [row[0] for row in category_query]
            num_of_obs = session.execute(f"SELECT COUNT(*) FROM ({dataset.load_query}) _subquery_").fetchone()[0]

            if cause_condition['categorical']:
                cause_predicate = (f"_subquery_.\"{cause_node.name}\" IN "
                                   f"({','.join(map(repr, cause_condition['values']))})")
            else:
                cause_predicate = (f"_subquery_.\"{cause_node.name}\" >= {repr(cause_condition['from_value'])} AND "
                                   f"_subquery_.\"{cause_node.name}\" < {repr(cause_condition['from_value'])}")

            probabilities = []
            for category in categories:
                if len(probabilities) == len(categories) - 1:  # Probabilities will sum to 1
                    probabilities.append(1 - sum(probabilities))
                else:
                    do_sql = f"SELECT " \
                             f"COUNT(*) AS group_count, " \
                             f"COUNT(CASE WHEN {cause_predicate} THEN 1 ELSE NULL END) AS marginal_count, " \
                             f"COUNT(CASE WHEN {cause_predicate} " \
                             f"AND _subquery_.\"{effect_node.name}\"={repr(category)} THEN 1 ELSE NULL END) " \
                             f"AS conditional_count FROM ({dataset.load_query}) _subquery_ "
                    if len(factor_nodes) > 0:
                        factor_str = ','.join(['_subquery_.\"' + n.name + '\"' for n in factor_nodes])
                        do_sql += f"GROUP BY {factor_str}"
                    do_query = session.execute(do_sql).fetchall()
                    group_counts, marg_counts, cond_counts = zip(*[(line[-3], line[-2], line[-1]) for line in do_query])

                    probability = sum([
                        (cond_count / marg_count) * (group_count / sum(group_counts))
                        for group_count, marg_count, cond_count in zip(group_counts, marg_counts, cond_counts)
                        if marg_count > 0
                    ])
                    probabilities.append(probability)

            bins = dict([(str(cat), round(num_of_obs * float(prob))) for cat, prob in zip(categories, probabilities)])

            return marshal(DiscreteDistributionSchema, {
                'node': effect_node,
                'dataset': dataset,
                'bins': bins
            })
        else:
            factor_str = (', ' + ', '.join([f'"{n.name}"' for n in factor_nodes])) if len(factor_nodes) > 0 else ''
            result = session.execute(f"SELECT \"{cause_node.name}\", \"{effect_node.name}\"{factor_str} "
                                     f"FROM ({dataset.load_query}) _subquery_").fetchall()
            arr = np.array([line for line in result])

            _, bin_edges = _custom_histogram(arr[:, 1])

            arr[:, 1:] = np.apply_along_axis(
                lambda c: np.digitize(c, _custom_histogram(c)[1][:-1]), 0, arr[:, 1:])
            df = pd.DataFrame(arr, columns=([cause_node.name] + [effect_node.name] + [f.name for f in factor_nodes]))

            probabilities = []
            for effect_bin in range(1, len(bin_edges) - 1):
                group_counts, marg_counts, cond_counts = [], [], []
                factor_grouping = df.groupby(df.columns[2:].tolist()) if len(df.columns) > 2 else [('', df)]
                for factor_group, factor_df in factor_grouping:
                    if cause_condition['categorical']:
                        cause_mask = factor_df[cause_node.name].isin(cause_condition['values'])
                    else:
                        cause_mask = ((factor_df[cause_node.name] >= cause_condition['from_value']) &
                                      (factor_df[cause_node.name] < cause_condition['to_value']))

                    group_counts.append(len(factor_df))
                    marg_counts.append(len(factor_df[cause_mask]))
                    cond_counts.append(len(factor_df[cause_mask & (factor_df[effect_node.name] == effect_bin)]))

                probability = sum([
                    (cond_count / marg_count) * (group_count / sum(group_counts))
                    for group_count, marg_count, cond_count in zip(group_counts, marg_counts, cond_counts)
                    if marg_count > 0
                ])
                probabilities.append(probability)
            probabilities.append(1 - sum(probabilities))

            bins = [round(len(df) * float(prob)) for prob in probabilities]
            return marshal(ContinuousDistributionSchema, {
                'node': effect_node,
                'dataset': dataset,
                'bins': bins,
                'bin_edges': bin_edges,
            })
