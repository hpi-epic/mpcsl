import time
from threading import Thread
import psutil

from src.db import db
from src.master.config import DAEMON_CYCLE_TIME
from src.models import Job, JobStatus


def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        process = psutil.Process(pid)
        return process.status() in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
    except psutil.NoSuchProcess:
        return False


class JobDaemon(Thread):
    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = app
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            time.sleep(DAEMON_CYCLE_TIME)

            with self.app.app_context():
                self.app.logger.info('Daemon cycle')
                self.app.logger.info(psutil.pids())

                jobs = db.session.query(Job).filter(Job.status == JobStatus.running)

                for job in jobs:
                    if not check_pid(job.pid):
                        self.app.logger.warning('Job ' + str(job.id) + ' failed')
                        job.status = JobStatus.error

                db.session.commit()

        with self.app.app_context():
            self.app.logger.warning('Daemon cancelled')
