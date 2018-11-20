from datetime.datetime import now

from flask import current_app, request
from flask_restful import Resource

from src import db
from src.models import Job, Result, Node, Edge, SepSet


class Results(Resource):

    def post(self):
        json = request.json
        job = Job.query.get(json['job_id'])

        result = Result(experiment=job.experiment, start_time=job.start_time,
                        end_time=now(), meta_results=json['meta_results'])
        db.session.add(result)

        node_list = json['node_list']
        node_mapping = {}
        for node_name in node_list:
            node = Node(name=node_name, result=result)
            node_mapping[node_name] = node
            db.session.add(node)

        edge_list = json['edge_list']
        for from_node, to_node in edge_list:
            edge = Edge(from_node=node_mapping[from_node], to_node=node_mapping[to_node],
                        result=result)
            db.session.add(edge)

        sepset_list = json['sepset_list']
        for sepset in sepset_list:
            sepset = SepSet(nodes=sepset['nodes'], statistic=sepset['statistic'],
                            level=sepset['level'], result=result)
            db.session.add(sepset)

        db.session.delete(job)
        current_app.logger.info('Result {} created'.format(result.id))
        db.session.commit()
        return 'OK'
