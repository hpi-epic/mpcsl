from datetime import datetime
import os
import signal

from flask import current_app
from flask_restful import Resource
from marshmallow import fields, Schema

from src.db import db
from src.master.helpers.io import marshal, load_data
from src.models import Job, JobSchema, ResultSchema, Edge, Node, Result
from src.models.job import JobStatus


class JobResource(Resource):
    def get(self, job_id):
        job = Job.query.get_or_404(job_id)

        return marshal(JobSchema, job)

    def delete(self, job_id):
        job = Job.query.get_or_404(job_id)

        os.kill(job.pid, signal.SIGTERM)

        job.status = JobStatus.killed
        db.session.commit()

        return marshal(JobSchema, job)


class JobListResource(Resource):
    def get(self):
        job = Job.query.all()

        return marshal(JobSchema, job, many=True)


class JobResultResource(Resource):
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

        db.session.delete(job)
        current_app.logger.info('Result {} created'.format(result.id))
        job.status = JobStatus.done
        db.session.commit()
        return marshal(ResultSchema, result)


class EdgeResultEndpointSchema(Schema):
    from_node = fields.String()
    to_node = fields.String()


class ResultEndpointSchema(Schema):
    meta_results = fields.Dict()
    node_list = fields.List(fields.String())
    edge_list = fields.Nested(EdgeResultEndpointSchema, many=True)
