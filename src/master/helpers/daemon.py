import psutil
import docker

from src.db import db
from src.master.config import DAEMON_CYCLE_TIME
from src.master.helpers.docker import get_client
from src.models import Job, JobStatus


def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        process = psutil.Process(pid)
        return process.status() in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
    except psutil.NoSuchProcess:
        return False


def start_job_daemon(app, socketio):
    client = get_client()
    while True:
        socketio.sleep(DAEMON_CYCLE_TIME)
        with app.app_context():
            app.logger.info('Daemon cycle')
            jobs = db.session.query(Job).filter(Job.status == JobStatus.running)

            for job in jobs:
                try:
                    container = client.containers.get(job.container_id)
                    if container.status == "exited":
                        app.logger.warning('Job ' + str(job.id) + ' failed')
                        job.status = JobStatus.error
                except docker.errors.NotFound:
                    app.logger.warning('Job ' + str(job.id) + ' disappeared')
                    job.status = JobStatus.error
            db.session.commit()
