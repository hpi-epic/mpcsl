import os
from datetime import datetime
from subprocess import Popen
from flask_restful_swagger_2 import swagger

from flask import current_app
from flask_restful import Resource, abort

from src.master.config import API_HOST, LOAD_SEPARATION_SET
from src.master.helpers.io import marshal, get_logfile_name
from src.db import db
from src.master.helpers.swagger import get_default_response
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
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Experiment']
    })
    def post(self, experiment_id):
        experiment = Experiment.query.get_or_404(experiment_id)

#        if not check_dataset_hash(experiment.dataset):
#            new_dataset = invalidate_dataset(experiment.dataset)
#            new_experiment = duplicate_experiment(experiment, new_dataset)
#            experiment = new_experiment

        algorithm = experiment.algorithm

        new_job = Job(experiment=experiment, start_time=datetime.now(),
                      status=JobStatus.running)
        db.session.add(new_job)
        db.session.flush()

        logfile = get_logfile_name(new_job.id)
        if os.path.isfile(logfile):
            # backup log files that are already existing
            renamed_logfile = f'{logfile[:-4]}_{datetime.now()}.log'
            try:
                os.rename(logfile, renamed_logfile)
            except OSError:
                current_app.logger.warn(f'Could not rename existing log file {logfile} to {renamed_logfile}')
        if algorithm.backend == 'R':
            params = []
            for k, v in experiment.parameters.items():
                params.append('--' + k)
                params.append(str(v))
            with open(logfile, 'a') as logfile:
                r_process = Popen([
                    'Rscript', 'src/master/executor/algorithms/r/' + algorithm.script_filename,
                    '-j', str(new_job.id),
                    '-d', str(experiment.dataset_id),
                    '--api_host', str(API_HOST),
                    '--send_sepsets', str(int(LOAD_SEPARATION_SET))
                ] + params, start_new_session=True, stdout=logfile, stderr=logfile)
            new_job.pid = r_process.pid
            db.session.commit()
        else:
            abort(501)
        return marshal(JobSchema, new_job)
