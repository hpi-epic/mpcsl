from codecs import getreader
from decimal import Decimal

import ijson
from flask import request
from flask_restful import Resource
from flask_restful_swagger_2 import swagger
from ijson.common import ObjectBuilder
from marshmallow import fields, Schema

from src.db import db
from src.master.config import RESULT_READ_BUFF_SIZE, LOAD_SEPARATION_SET, RESULT_WRITE_BUFF_SIZE
from src.master.helpers.io import marshal, InvalidInputData
from src.master.helpers.swagger import get_default_response
from src.models import Edge, Experiment, ExperimentJob, Job, Result, Sepset
from src.models.base import SwaggerMixin
from src.models.experiment_job import ExperimentJobSchema


class ExperimentJobListResource(Resource):
    @swagger.doc({
        'description': 'Returns all jobs of one specific experiment ordered by their start time',
        'parameters': [
            {
                'name': 'experiment_id',
                'description': 'Experiment identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(ExperimentJobSchema.get_swagger().array()),
        'tags': ['Job', 'Experiment']
    })
    def get(self, experiment_id):
        Experiment.query.get_or_404(experiment_id)
        jobs = ExperimentJob.query \
            .filter(ExperimentJob.experiment_id == experiment_id) \
            .order_by(Job.start_time.desc())

        return marshal(ExperimentJobSchema, jobs, many=True)


class EdgeResultEndpointSchema(Schema, SwaggerMixin):
    from_node = fields.String()
    to_node = fields.String()


class SepsetResultEndpointSchema(Schema, SwaggerMixin):
    from_node = fields.String()
    to_node = fields.String()
    nodes = fields.List(fields.String())
    level = fields.Integer()
    statistic = fields.Float()


class ExperimentResultEndpointSchema(Schema, SwaggerMixin):
    meta_results = fields.Dict()
    node_list = fields.List(fields.String())
    edge_list = fields.Nested(EdgeResultEndpointSchema, many=True)
    sepset_list = fields.Nested(SepsetResultEndpointSchema, many=True)


def ijson_parse_items(file, prefixes):
    """
    An iterator returning native Python objects constructed from the events
    under a list of given prefixes.
    """
    prefixed_events = iter(ijson.parse(getreader('utf-8')(file), buf_size=RESULT_READ_BUFF_SIZE))
    try:
        while True:
            current, event, value = next(prefixed_events)
            matches = [prefix for prefix in prefixes if current == prefix]
            if len(matches) > 0:
                prefix = matches[0]
                if event in ('start_map', 'start_array'):
                    builder = ObjectBuilder()
                    end_event = event.replace('start', 'end')
                    while (current, event) != (prefix, end_event):
                        builder.event(event, value)
                        current, event, value = next(prefixed_events)
                    yield prefix, builder.value
                else:
                    yield prefix, value
    except StopIteration:
        pass


def process_experiment_job_result(job: Job, result: Result):
    node_ids = {n.id for n in job.experiment.dataset.nodes}

    result_elements = ijson_parse_items(
        request.stream,
        ['meta_results', 'edge_list.item', 'sepset_list.item', 'execution_time', 'dataset_loading_time']
    )
    edges = []
    sepsets = []
    for prefix, element in result_elements:
        if prefix == 'meta_results':
            for key, val in element.items():
                if isinstance(val, Decimal):
                    element[key] = float(val)
            result.meta_results = element
        elif prefix == 'edge_list.item':
            if element['from_node'] not in node_ids or element['to_node'] not in node_ids:
                raise InvalidInputData('Invalid Node ID')
            edge = Edge(
                from_node_id=element['from_node'],
                to_node_id=element['to_node'],
                result_id=result.id,
                weight=element.get('weight', None)
            )
            edges.append(edge)

            if len(edges) > RESULT_WRITE_BUFF_SIZE:
                db.session.bulk_save_objects(edges)
                edges = []

        elif prefix == 'sepset_list.item' and LOAD_SEPARATION_SET:
            if element['from_node'] not in node_ids or element['to_node'] not in node_ids:
                raise InvalidInputData('Invalid Node ID')
            sepset = Sepset(
                statistic=element['statistic'],
                level=element['level'],
                from_node_id=element['from_node'],
                to_node_id=element['to_node'],
                result_id=result.id
            )
            sepsets.append(sepset)

            if len(sepsets) > RESULT_WRITE_BUFF_SIZE:
                db.session.bulk_save_objects(sepsets)
                sepsets = []

        elif prefix == 'execution_time':
            result.execution_time = element

        elif prefix == 'dataset_loading_time':
            result.dataset_loading_time = element

    if len(edges) > 0:
        db.session.bulk_save_objects(edges)
    if len(sepsets) > 0:
        db.session.bulk_save_objects(sepsets)
