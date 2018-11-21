from datetime import datetime

from flask import current_app
from flask_restful import Resource
from marshmallow import Schema, fields

from src.db import db
from src.master.helpers.io import marshal, load_data
from src.models import Job, Result, ResultSchema, Node, Edge


class ResultListResource(Resource):

    def post(self):
        json = load_data(ResultEndpointSchema)
        job = Job.query.get_or_404(json['job_id'])

        result = Result(experiment=job.experiment, start_time=job.start_time,
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
        db.session.commit()
        return marshal(ResultSchema, result)


class EdgeResultSchema(Schema):
    from_node = fields.String()
    to_node = fields.String()


class ResultEndpointSchema(Schema):
    job_id = fields.Integer()
    meta_results = fields.Dict()
    node_list = fields.List(fields.String())
    edge_list = fields.Nested(EdgeResultSchema, many=True)
