import os
import signal
from datetime import datetime
from subprocess import Popen, PIPE

from flask import current_app, send_file, Response
from flask_restful import Resource, abort, reqparse
from flask_restful_swagger_2 import swagger
from marshmallow import fields, Schema

from src.db import db
from src.master.helpers.io import marshal, load_data, remove_logs, get_logfile_name
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
        json = load_data(ResultEndpointSchema)
        job = Job.query.get_or_404(job_id)

        result = Result(job=job, start_time=job.start_time,
                        end_time=datetime.now(),
                        meta_results=json['meta_results'])
        db.session.add(result)

        node_list = json['node_list']
        node_mapping = {}
        for node_name in node_list:
            node = Node.query().filter_by(dataset_id=job.experiment.dataset_id, name=node_name)
            node_mapping[node_name] = node

        edge_list = json['edge_list']
        for edge in edge_list:
            edge = Edge(from_node=node_mapping[edge['from_node']], to_node=node_mapping[edge['to_node']],
                        result=result)
            db.session.add(edge)

        sepset_list = json['sepset_list']
        for sepset in sepset_list:
            sepset = Sepset(node_names=sepset['nodes'], statistic=sepset['statistic'],
                            level=sepset['level'], from_node=node_mapping[sepset['from_node']],
                            to_node=node_mapping[sepset['to_node']], result=result)
            db.session.add(sepset)

        current_app.logger.info('Result {} created'.format(result.id))
        job.status = JobStatus.done
        job.result_id = result.id
        db.session.commit()
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
