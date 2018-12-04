from datetime import datetime
from subprocess import Popen

from flask import current_app
from flask_restful import Resource

from src.master.helpers.io import marshal
from src.db import db
from src.models.job import Job, JobSchema
from src.models.experiment import Experiment


class ExecutorResource(Resource):

    def get(self, experiment_id):
        current_app.logger.info('Got request')
        experiment = Experiment.query.get_or_404(experiment_id)

        new_job = Job(experiment=experiment, start_time=datetime.now())
        db.session.add(new_job)
        db.session.flush()

        params = []
        for k, v in experiment.parameters.items():
            params.append('--' + k)
            params.append(str(v))

        r_process = Popen([
            'Rscript', 'src/master/executor/algorithms/r/pcalg.r',
            '-j', str(new_job.id),
            '-d', str(experiment.dataset_id)
        ] + params)

        new_job.pid = r_process.pid
        db.session.commit()
        return marshal(JobSchema, new_job)
