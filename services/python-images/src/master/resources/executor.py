from datetime import datetime

from flask import request
from flask_restful import Resource, abort
from flask_restful_swagger_2 import swagger

from src.db import db
from src.master.helpers.database import check_dataset_hash
from src.master.helpers.io import marshal
from src.models import JobSchema, JobStatus, Experiment, ExperimentJob


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
        gpus = None
        body = request.json
        enforce_cpus = True
        if body:
            node_hostname = body.get("node")
            runs = body.get('runs')
            parallel = body.get('parallel')
            gpus = body.get('gpus')
            enforce_cpus = body.get('enforce_cpus')
        for i in range(runs):
            new_experiment_job = ExperimentJob(start_time=datetime.now(), status=JobStatus.waiting,
                                               node_hostname=node_hostname, experiment=experiment,
                                               parallel=parallel, gpus=gpus, enforce_cpus=enforce_cpus)

            db.session.add(new_experiment_job)
        db.session.commit()

        return marshal(JobSchema, new_experiment_job)
