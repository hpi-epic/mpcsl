import os
import signal

from flask_restful import Resource

from src.db import db
from src.master.helpers.io import marshal
from src.models import Job, JobSchema


class JobResource(Resource):
    def get(self, job_id):
        job = Job.query.get_or_404(job_id)

        return marshal(JobSchema, job)

    def delete(self, job_id):
        job = Job.query.get_or_404(job_id)

        os.kill(job.pid, signal.SIGTERM)
        
        data = marshal(JobSchema, job)
        db.session.delete(job)

        db.session.commit()

        return data


class JobListResource(Resource):
    def get(self):
        job = Job.query.all()

        return marshal(JobSchema, job, many=True)
