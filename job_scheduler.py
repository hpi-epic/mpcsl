import asyncio
import logging

from aiohttp import web, ClientSession

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.master.config import DAEMON_CYCLE_TIME, SQLALCHEMY_DATABASE_URI, API_HOST
from src.models import Job, JobStatus, Experiment
from src.jobscheduler.kubernetes_helper import createJob, kube_cleanup_finished_jobs, checkRunningJob


logging.basicConfig(level=logging.INFO)


async def health_check(request):
    return web.Response(text="ok")


async def post_job_change(session, job_id, status):
    url = "http://" + API_HOST + "/api/job/" + str(job_id)
    logging.info("Emit change to %s", url)
    async with session.post(url, json={'status': status}) as resp:
        resp.raise_for_status()
        logging.info(await resp.text())


async def start_waiting_jobs(session: Session):
    jobs = session.query(Job).filter(Job.status == JobStatus.waiting)
    for job in jobs:
        try:
            experiment = session.query(Experiment).get(job.experiment_id)
            k8s_job_name = await createJob(job, experiment)
            if isinstance(k8s_job_name, str):
                job.container_id = k8s_job_name
                job.status = JobStatus.running
                asyncio.create_task(post_job_change(ClientSession(), job.id, job.status))
                session.commit()
        except Exception as e:
            logging.error(str(e))
            job.status = JobStatus.error
            asyncio.create_task(post_job_change(ClientSession(), job.id, job.status))
            session.commit()


async def kill_errored_jobs(session: Session):
    jobs = session.query(Job).filter(Job.status == JobStatus.running)
    for job in jobs:
        try:
            crashed = await checkRunningJob(job)
            if crashed:
                job.status = JobStatus.error
                asyncio.create_task(post_job_change(ClientSession(), job.id, job.status))
                session.commit()
        except Exception as e:
            logging.error(str(e))

async def start_job_scheduler():
    logging.info("Starting Job Scheduler")
    engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
    Session = sessionmaker(bind=engine)
    while True:
        try:
            await asyncio.sleep(DAEMON_CYCLE_TIME)
            await asyncio.gather(start_waiting_jobs(Session()), kill_errored_jobs(Session()))
            kube_cleanup_finished_jobs()
        except Exception as e:
            logging.error(str(e))


async def start_background_tasks(app):
    app['scheduler'] = asyncio.create_task(start_job_scheduler())


async def cleanup_background_tasks(app):
    app['scheduler'].cancel()
    await app['scheduler']


def main():
    app = web.Application()
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    app.add_routes([web.get('/', health_check)])
    web.run_app(app)


if __name__ == "__main__":
    main()
