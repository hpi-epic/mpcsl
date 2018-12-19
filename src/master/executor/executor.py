from datetime import datetime
from subprocess import Popen
from flask_restful_swagger_2 import swagger

from flask import current_app
from flask_restful import Resource

from src.master.config import API_HOST
from src.master.helpers.io import marshal
from src.db import db
from src.master.helpers.swagger import get_default_response
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
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Experiment']
    })
    def post(self, experiment_id):
        current_app.logger.info('Got request')
        experiment = Experiment.query.get_or_404(experiment_id)

        new_job = Job(experiment=experiment, start_time=datetime.now(), status=JobStatus.running)
        db.session.add(new_job)
        db.session.flush()

        params = []
        for k, v in experiment.parameters.items():
            params.append('--' + k)
            params.append(str(v))

        r_process = Popen([
            'Rscript', 'src/master/executor/algorithms/r/pcalg.r',
            '-j', str(new_job.id),
            '-d', str(experiment.dataset_id),
            '--api_host', str(API_HOST)
        ] + params)

        new_job.pid = r_process.pid
        db.session.commit()
        return marshal(JobSchema, new_job)
