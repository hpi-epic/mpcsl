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
    bins = fields.List(fields.Int())
    bin_edges = fields.List(fields.Float())


class MarginalDistributionResource(Resource):
    @swagger.doc({
        'description': 'Returns the marginal distribution of an attribute as histogram values',
        'parameters': [
            {
                'name': 'node_id',
                'description': 'Node identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(DistributionSchema.get_swagger()),
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

        result = session.execute(f'SELECT {node.name} FROM ({dataset.load_query})')
        values = [line[0] for line in result]
        hist, bin_edges = np.histogram(values, bins='auto', density=False)

        return marshal(DistributionSchema, {
            'node': node,
            'dataset': dataset,
            'bins': hist,
            'bin_edges': bin_edges
        })
