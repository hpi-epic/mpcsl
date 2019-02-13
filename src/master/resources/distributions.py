import numpy as np
from flask_restful_swagger_2 import swagger
from flask_restful import Resource, abort
from marshmallow import fields

from src.db import db
from src.master.db import data_source_connections
from src.master.helpers.io import marshal
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
    bins = fields.Dict(values=fields.Int(), keys=fields.String())
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
