from datetime import datetime
from subprocess import Popen

from flask_restful import Resource

from src.master.helpers.io import marshal
from src.db import db
from src.models.job import Job, JobSchema
from src.models.experiment import Experiment


class Executor(Resource):

    def get(self, experiment_id):
        experiment = Experiment.query.get_or_404(experiment_id)

        new_job = Job(experiment=experiment, start_time=datetime.now())

        db.session.add(new_job)

        db.session.flush()

        r_process = Popen(['Rscript', 'src/master/executor/algorithms/r/pcalg.r', '-j', str(1),#new_job.id,
                           '-d', str(1),#str(experiment.dataset_id),
                           '-a', str(0.9),#str(experiment.alpha),
                           '-c', str(1)])#str(experiment.cores)])

        new_job.pid = r_process.pid

        return marshal(JobSchema, new_job)
