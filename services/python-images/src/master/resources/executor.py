from datetime import datetime

from flask import request
from flask_restful import Resource, abort
from flask_restful_swagger_2 import swagger

from src.db import db
from src.master.helpers.database import check_dataset_hash
from src.master.helpers.io import marshal
from src.models import Job, JobSchema, JobStatus, Experiment


class ExecutorResource(Resource):

    @swagger.doc({
        'description': 'Starts a new job for this experiment',
        'parameters': [
            {
                'name': 'experiment_id',
                'description': 'Experiment identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': {
            '200': {
                'description': 'Success',
            },
            '400': {
                'description': 'Invalid input data'
            },
            '409': {
                'description': 'Conflict: The underlying data has been modified'
            },
            '500': {
                'description': 'Internal server error'
            }
        },
        'tags': ['Experiment']
    })
    def post(self, experiment_id):
        experiment = Experiment.query.get_or_404(experiment_id)

        if not check_dataset_hash(experiment.dataset):
            abort(409, message='The underlying data has been modified')
        node_hostname = None
        parallel = True
        runs = 1
        body = request.json
        if body:
            node_hostname = body.get("node")
            runs = request.json.get('runs')
            parallel = request.json.get('parallel')
        for i in range(runs):
            new_job = Job(experiment=experiment, start_time=datetime.now(),
                          status=JobStatus.waiting, node_hostname=node_hostname, parallel=parallel)
            db.session.add(new_job)
        db.session.commit()

        return marshal(JobSchema, new_job)
