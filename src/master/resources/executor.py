from datetime import datetime

import docker
from flask_restful_swagger_2 import swagger

from flask_restful import Resource, abort
from werkzeug.exceptions import BadRequest

from src.master.config import API_HOST, LOAD_SEPARATION_SET, DOCKER_EXECUTION_NETWORK
from src.master.helpers.docker import get_client
from src.master.helpers.io import marshal
from src.db import db
from src.models import Job, JobSchema, JobStatus, Experiment
from src.master.helpers.database import check_dataset_hash


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

        algorithm = experiment.algorithm

        new_job = Job(experiment=experiment, start_time=datetime.now(),
                      status=JobStatus.running)
        db.session.add(new_job)
        db.session.flush()

        params = [
            '-j', str(new_job.id),
            '-d', str(experiment.dataset_id),
            '--api_host', str(API_HOST),
            '--send_sepsets', str(int(LOAD_SEPARATION_SET))
        ]
        for k, v in experiment.parameters.items():
            params.append('--' + k)
            params.append(str(v))

        client = get_client()
        command = algorithm.script_filename + " " + " ".join(params)
        try:
            container = client.containers.run(
                algorithm.docker_image,
                command,
                detach=True,
                network=DOCKER_EXECUTION_NETWORK if DOCKER_EXECUTION_NETWORK else None,
                **algorithm.docker_parameters
            )
        except docker.errors.ImageNotFound:
            raise BadRequest(
                f'Image {algorithm.docker_image} not found. Did you build '
                f'the containers available in /src/executionenvironments?')

        new_job.container_id = container.id
        db.session.commit()

        return marshal(JobSchema, new_job)
