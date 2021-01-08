
import requests
from flask import current_app, Response, request
from flask_restful import Resource, abort, reqparse
from flask_restful_swagger_2 import swagger

from src.db import db
from src.master.config import SCHEDULER_HOST
from src.master.helpers.io import marshal
from src.master.helpers.socketio_events import job_status_change
from src.master.helpers.swagger import get_default_response, oneOf
from src.models import Job, JobSchema, Result, ResultSchema
from src.models.job import JobStatus
from datetime import datetime
from src.master.resources.experiment_jobs import ExperimentResultEndpointSchema, process_experiment_job_result


def kill_container(container):
    requests.post(f'http://{SCHEDULER_HOST}/api/delete/{container}')


class JobResource(Resource):
    @swagger.doc({
        'description': 'Returns a single job',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job']
    })
    def get(self, job_id):
        job = Job.query.get_or_404(job_id)

        return marshal(JobSchema, job)

    @swagger.doc({
        'description': 'Updates the status of a running job to "error"',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job', 'Executor']
    })
    def put(self, job_id):
        job = Job.query.get_or_404(job_id)
        if job.status != JobStatus.running:
            abort(400)

        current_app.logger.info('An error occurred in Job {}'.format(job.id))
        job.status = JobStatus.error
        db.session.commit()
        return marshal(JobSchema, job)

    @swagger.doc({
        'description': 'Cancels a single job if running, killing the process. Otherwise sets it to hidden.',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job']
    })
    def delete(self, job_id):
        job: Job = Job.query.get_or_404(job_id)

        if job.status == JobStatus.running:
            kill_container(job.container_id)
            job.status = JobStatus.cancelled
        else:
            job.status = JobStatus.hidden
        db.session.commit()
        return marshal(JobSchema, job)

    @swagger.doc({
        'description': 'Updates the status of a running job to "error"',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger()),
        'tags': ['Job', 'Executor']
    })
    def post(self, job_id):
        job: Job = Job.query.get_or_404(job_id)
        content = request.json
        error_code = content['error_code']
        if error_code is not None:
            error_code = int(error_code)
        job_status_change(job, error_code)
        return "ok"


class JobListResource(Resource):
    @swagger.doc({
        'description': 'Returns all jobs',
        'parameters': [
            {
                'name': 'show_hidden',
                'description': 'Pass show_hidden=1 to display also hidden jobs',
                'in': 'query',
                'type': 'integer',
                'enum': [0, 1],
                'default': 0
            }
        ],
        'responses': get_default_response(JobSchema.get_swagger().array()),
        'tags': ['Job']
    })
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('show_hidden', required=False, type=int, store_missing=False)
        show_hidden = parser.parse_args().get('show_hidden', 0) == 1

        jobs = Job.query.all() if show_hidden else Job.query.filter(Job.status != JobStatus.hidden)

        return marshal(JobSchema, jobs, many=True)


class JobLogsResource(Resource):
    @swagger.doc({
        'description': 'Get the log output (stdout/stderr) for a given job',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'offset',
                'description': 'Output logs starting with line n',
                'in': 'query',
                'type': 'integer',
            },
            {
                'name': 'last',
                'description': 'Output only the last n lines',
                'in': 'query',
                'type': 'integer'
            }
        ],
        'responses': {
            '200': {
                'description': 'Success',
            },
            '404': {
                'description': 'Log not found'
            },
            '500': {
                'description': 'Internal server error'
            }
        },
        'produces': ['text/plain'],
        'tags': ['Executor', 'Job']
    })
    def get(self, job_id):
        job: Job = Job.query.get_or_404(job_id)

        parser = reqparse.RequestParser()
        parser.add_argument('offset', required=False, type=int, store_missing=False)
        parser.add_argument('last', required=False, type=int, store_missing=False)
        args = parser.parse_args()
        offset = args.get('offset', 0)
        last = args.get('last', 0)
        log = job.log

        if log is None:
            log = requests.get(f'http://{SCHEDULER_HOST}/api/log/{job.id}').content.decode()

        log = log.split('\n')
        if offset > 0 and last == 0:
            log = log[offset:]
        if offset == 0 and last > 0:
            log = log[-last:]
        log = "\n".join(log)

        return Response(log, mimetype='text/plain')


class JobResultResource(Resource):
    @swagger.doc({
        'description': 'Stores the results of job execution. Job is marked as done.',
        'parameters': [
            {
                'name': 'job_id',
                'description': 'Job identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'result',
                'description': 'Result data',
                'in': 'body',
                'schema': oneOf([ExperimentResultEndpointSchema]).get_swagger(True)
            }
        ],
        'responses': get_default_response(ResultSchema.get_swagger()),
        'tags': ['Executor']
    })
    def post(self, job_id):
        job = Job.query.get_or_404(job_id)
        end_time = datetime.now()
        result = Result(job=job, start_time=job.start_time,
                        end_time=end_time,)
        db.session.add(result)
        db.session.flush()

        current_app.logger.info('Result {} is in creation for job {}'.format(result.id, job.id))
        process_experiment_job_result(job, result)

        job.status = JobStatus.done
        job.result_id = result.id
        job.end_time = end_time
        db.session.commit()
        current_app.logger.info('Result {} created for job {}'.format(result.id, job.id))

        return marshal(ResultSchema, result)
