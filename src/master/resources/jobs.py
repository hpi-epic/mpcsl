import os
import signal
from codecs import getreader
from datetime import datetime
from decimal import Decimal
from subprocess import Popen, PIPE

from flask import current_app, send_file, Response, request
from flask_restful import Resource, abort, reqparse
from flask_restful_swagger_2 import swagger
from marshmallow import fields, Schema
import ijson
from ijson.common import ObjectBuilder

from src.db import db
from src.master.config import RESULT_READ_BUFF_SIZE, LOAD_SEPARATION_SET, RESULT_WRITE_BUFF_SIZE
from src.master.helpers.io import marshal, remove_logs, get_logfile_name, InvalidInputData
from src.master.helpers.swagger import get_default_response
from src.models import Job, JobSchema, ResultSchema, Edge, Node, Result, Sepset, Experiment
from src.models.base import SwaggerMixin
from src.models.job import JobStatus


class JobResource(Resource):
    @swagger.doc({
        'description': 'Returns a single job',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job']
    })
    def get(self, job_id):
        job = Job.query.get_or_404(job_id)

        return marshal(JobSchema, job)

    @swagger.doc({
        'description': 'Updates the status of a running job to "error"',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job', 'Executor']
    })
    def put(self, job_id):
        job = Job.query.get_or_404(job_id)
        if job.status != JobStatus.running:
            abort(400)

        current_app.logger.info('An error occurred in Job {}'.format(job.id))
        job.status = JobStatus.error
        db.session.commit()

        return marshal(JobSchema, job)

    @swagger.doc({
        'description': 'Cancels a single job if running, killing the process. Otherwise sets it to hidden.',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job']
    })
    def delete(self, job_id):
        job = Job.query.get_or_404(job_id)

        if job.status == JobStatus.running:
            try:
                os.killpg(os.getpgid(job.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            job.status = JobStatus.cancelled
        else:
            job.status = JobStatus.hidden
        db.session.commit()

        return marshal(JobSchema, job)


class JobListResource(Resource):
    @swagger.doc({
        'description': 'Returns all jobs',
        'parameters': [
            {
                'name': 'show_hidden',
                'description': 'Pass show_hidden=1 to display also hidden jobs',
                'in': 'query',
                'type': 'integer',
                'enum': [0, 1],
                'default': 0
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger().array()),
        'tags': ['Job']
    })
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('show_hidden', required=False, type=int, store_missing=False)
        show_hidden = parser.parse_args().get('show_hidden', 0) == 1

        jobs = Job.query.all() if show_hidden else Job.query.filter(Job.status != JobStatus.hidden)

        return marshal(JobSchema, jobs, many=True)


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
        'responses': get_default_response(JobSchema.get_swagger().array()),
        'tags': ['Job', 'Experiment']
    })
    def get(self, experiment_id):
        Experiment.query.get_or_404(experiment_id)
        jobs = Job.query\
            .filter(Job.experiment_id == experiment_id)\
            .order_by(Job.start_time.desc())

        return marshal(JobSchema, jobs, many=True)


class EdgeResultEndpointSchema(Schema, SwaggerMixin):
    from_node = fields.String()
    to_node = fields.String()


class SepsetResultEndpointSchema(Schema, SwaggerMixin):
    from_node = fields.String()
    to_node = fields.String()
    nodes = fields.List(fields.String())
    level = fields.Integer()
    statistic = fields.Float()


class ResultEndpointSchema(Schema, SwaggerMixin):
    meta_results = fields.Dict()
    node_list = fields.List(fields.String())
    edge_list = fields.Nested(EdgeResultEndpointSchema, many=True)
    sepset_list = fields.Nested(SepsetResultEndpointSchema, many=True)


def ijson_parse_items(file, prefixes):
    '''
    An iterator returning native Python objects constructed from the events
    under a list of given prefixes.
    '''
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


class JobResultResource(Resource):
    @swagger.doc({
        'description': 'Stores the results of job execution. Job is marked as done.',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'result',
                'description': 'Result data',
                'in': 'body',
                'schema': ResultEndpointSchema.get_swagger(True)
            }
        ],
        'responses': get_default_response(ResultSchema.get_swagger()),
        'tags': ['Executor']
    })
    def post(self, job_id):
        job = Job.query.get_or_404(job_id)
        result = Result(job=job, start_time=job.start_time,
                        end_time=datetime.now())
        db.session.add(result)
        db.session.flush()

        current_app.logger.info('Result {} is in creation for job {}'.format(result.id, job.id))

        node_mapping = {}

        result_elements = ijson_parse_items(
            request.stream,
            ['meta_results', 'node_list.item', 'edge_list.item', 'sepset_list.item']
        )
        edges = []
        sepsets = []
        flushed_nodes = False
        for prefix, element in result_elements:
            if prefix == 'meta_results':
                for key, val in element.items():
                    if isinstance(val, Decimal):
                        element[key] = float(val)
                result.meta_results = element
            if prefix == 'node_list.item':
                node = Node.query.filter_by(dataset_id=job.experiment.dataset_id, name=element).first()
                node_mapping[element] = node
            if prefix in ['edge_list.item', 'sepset_list.item']:
                if len(node_mapping) == 0:
                    current_app.logger.warning(
                        'Result {} for job {} could not be stored. Node list was not first.'.format(result.id, job.id)
                    )
                    raise InvalidInputData({'node_list': ['has to be sent first!']})
                if flushed_nodes is False:
                    db.session.flush()
                    flushed_nodes = True
                    current_app.logger.info('Flushing {} nodes'.format(len(node_mapping)))
            if prefix == 'edge_list.item':
                edge = Edge(
                    from_node_id=node_mapping[element['from_node']].id,
                    to_node_id=node_mapping[element['to_node']].id,
                    result_id=result.id,
                    weight=element.get('weight', None)
                )
                edges.append(edge)

                if len(edges) > RESULT_WRITE_BUFF_SIZE:
                    db.session.bulk_save_objects(edges)
                    edges = []

            if prefix == 'sepset_list.item' and LOAD_SEPARATION_SET:
                sepset = Sepset(
                    statistic=element['statistic'],
                    level=element['level'],
                    from_node_id=node_mapping[element['from_node']].id,
                    to_node_id=node_mapping[element['to_node']].id,
                    result_id=result.id
                )
                sepsets.append(sepset)

                if len(sepsets) > RESULT_WRITE_BUFF_SIZE:
                    db.session.bulk_save_objects(sepsets)
                    sepsets = []

        job.status = JobStatus.done
        if len(edges) > 0:
            db.session.bulk_save_objects(edges)
        if len(sepsets) > 0:
            db.session.bulk_save_objects(sepsets)
        job.result_id = result.id
        db.session.commit()

        current_app.logger.info('Result {} created for job {}'.format(result.id, job.id))

        return marshal(ResultSchema, result)


class JobLogsResource(Resource):
    @swagger.doc({
        'description': 'Get the log output (stdout/stderr) for a given job',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'offset',
                'description': 'Output logs starting with line OFFSET',
                'in': 'query',
                'type': 'integer',
            },
            {
                'name': 'limit',
                'description': 'Output only the last LIMIT lines',
                'in': 'query',
                'type': 'integer'
            }
        ],
        'responses': {
            '200': {
                'description': 'Success',
            },
            '404': {
                'description': 'Log not found'
            },
            '500': {
                'description': 'Internal server error'
            }
        },
        'produces': ['text/plain'],
        'tags': ['Executor', 'Job']
    })
    def get(self, job_id):
        job = Job.query.get_or_404(job_id)
        logfile = get_logfile_name(job.id)
        if not os.path.isfile(logfile):
            abort(404)

        parser = reqparse.RequestParser()
        parser.add_argument('offset', required=False, type=int, store_missing=False)
        parser.add_argument('limit', required=False, type=int, store_missing=False)
        args = parser.parse_args()
        offset = args.get('offset', 0)
        limit = args.get('limit', 0)

        def run(cmd):
            p = Popen(cmd.split(), stdout=PIPE, universal_newlines=True)
            stdout, stderr = p.communicate()
            return stdout

        if offset > 0 and limit == 0:
            log = run(f'tail --lines +{offset} {logfile}')  # return all lines starting from 'offset'
            return Response(log, mimetype='text/plain')
        elif limit > 0 and offset == 0:
            command = f'tail --lines {limit} {logfile}'  # return last 'limit' lines
            return Response(run(command), mimetype='text/plain')
        elif limit > 0 and offset > 0:
            abort(501)
        else:
            return send_file(logfile, mimetype='text/plain')

    @swagger.doc({
        'description': 'Removes the associated log files',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job']
    })
    def delete(self, job_id):
        job = Job.query.get_or_404(job_id)

        if job.status == JobStatus.running:
            abort(403)
        else:
            remove_logs(job_id)
        return marshal(JobSchema, job)
