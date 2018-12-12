from datetime import datetime
import os
import signal
from flask_restful_swagger_2 import swagger

from flask import current_app
from flask_restful import Resource
from marshmallow import fields, Schema

from src.db import db
from src.master.helpers.io import marshal, load_data
from src.master.helpers.swagger import get_default_response
from src.models import Job, JobSchema, ResultSchema, Edge, Node, Result
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
        'responses': get_default_response(JobSchema.get_swagger())
    })
    def get(self, job_id):
        job = Job.query.get_or_404(job_id)

        return marshal(JobSchema, job)

    @swagger.doc({
        'description': 'Deletes a single job, this also kills the process',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger())
    })
    def delete(self, job_id):
        job = Job.query.get_or_404(job_id)

        os.kill(job.pid, signal.SIGTERM)

        job.status = JobStatus.cancelled
        db.session.commit()

        return marshal(JobSchema, job)


class JobListResource(Resource):
    @swagger.doc({
        'description': 'Returns all jobs',
        'responses': get_default_response(JobSchema.get_swagger().array())
    })
    def get(self):
        job = Job.query.all()

        return marshal(JobSchema, job, many=True)


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
        'responses': get_default_response(JobSchema.get_swagger().array())
    })
    def get(self, experiment_id):
        job = Job.query\
            .filter(Job.experiment_id == experiment_id)\
            .order_by(Job.start_time.desc())

        return marshal(JobSchema, job, many=True)


class EdgeResultEndpointSchema(Schema, SwaggerMixin):
    from_node = fields.String()
    to_node = fields.String()


class ResultEndpointSchema(Schema, SwaggerMixin):
    meta_results = fields.Dict()
    node_list = fields.List(fields.String())
    edge_list = fields.Nested(EdgeResultEndpointSchema, many=True)


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
        'responses': get_default_response(ResultSchema.get_swagger())
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
            node = Node(name=node_name, result=result)
            node_mapping[node_name] = node
            db.session.add(node)

        edge_list = json['edge_list']
        for edge in edge_list:
            edge = Edge(from_node=node_mapping[edge['from_node']], to_node=node_mapping[edge['to_node']],
                        result=result)
            db.session.add(edge)

        # sepset_list = json['sepset_list']
        # for sepset in sepset_list:
        #     sepset = SepSet(nodes=sepset['nodes'], statistic=sepset['statistic'],
        #                     level=sepset['level'], result=result)
        #     db.session.add(sepset)

        current_app.logger.info('Result {} created'.format(result.id))
        job.status = JobStatus.done
        db.session.commit()
        return marshal(ResultSchema, result)
