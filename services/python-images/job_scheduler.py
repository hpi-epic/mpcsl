import asyncio
import logging

from aiohttp import web

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.master.config import DAEMON_CYCLE_TIME, SQLALCHEMY_DATABASE_URI, PORT
from src.models import Job, JobStatus, Experiment, JobErrorCode
from src.jobscheduler.kubernetes_helper import create_experiment_job, kube_cleanup_finished_jobs, get_node_list
from src.jobscheduler.kubernetes_helper import check_running_job, get_pod_log, delete_job_and_pods
from src.jobscheduler.kubernetes_helper import EMPTY_LOGS
from src.jobscheduler.backend_requests import post_job_change

logging.basicConfig(level=logging.INFO)

engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
Session = sessionmaker(bind=engine)

routes = web.RouteTableDef()


@routes.get('/api/log/{job_id}')
async def get_job_log(request):
    job_id = request.match_info['job_id']
    log = await get_pod_log(job_id)
    if log is None:
        raise web.HTTPNotFound()
    return web.Response(text=log)


@routes.post('/api/delete/{job_name}')
async def delete_job(request):
    job_name = request.match_info['job_name']
    await delete_job_and_pods(job_name)
    return web.HTTPSuccessful()


@routes.get('/api/nodes')
async def get_nodes(request):
    nodes = await get_node_list()
    return web.json_response(nodes)


@routes.get('/')
async def health_check(request):
    return web.Response(text="ok")


async def start_waiting_jobs(session):
    logging.info("--- Check waiting jobs routine ---")
    jobs = session.query(Job).filter(Job.status == JobStatus.waiting)
    for job in jobs:
        try:
            running_jobs = session.query(Job).filter(Job.status == JobStatus.running)
            running_jobs_parallel = all([rj.parallel for rj in running_jobs])
            if (not running_jobs.all()) or (running_jobs_parallel and job.parallel):
                    experiment = session.query(Experiment).get(job.experiment_id)
                    k8s_job_name = await create_experiment_job(job, experiment)
                    if isinstance(k8s_job_name, str):
                        job.container_id = k8s_job_name
                        job.status = JobStatus.running
                        asyncio.create_task(post_job_change(job.id, None))
                        session.commit()
        except Exception as e:
            logging.error(str(e))
            job.status = JobStatus.error
            job.error_code = JobErrorCode.UNKNOWN
            session.commit()
            asyncio.create_task(post_job_change(job.id, job.error_code))


async def kill_errored_jobs(session):
    logging.info("--- Kill errored jobs routine ---")
    jobs = session.query(Job).filter(Job.status == JobStatus.running)
    for job in jobs:
        try:
            crashed = await check_running_job(job)
            if crashed:
                job: Job
                job.status = JobStatus.error
                job.error_code = JobErrorCode.UNKNOWN
                if job.log is None:
                    job.log = EMPTY_LOGS
                session.commit()
                asyncio.create_task(post_job_change(job.id, job.error_code))
        except Exception as e:
            logging.error(str(e))


async def loop_async_with_session(fnc):
    while True:
        try:
            await asyncio.sleep(DAEMON_CYCLE_TIME)
            await fnc(Session())
        except Exception as e:
            logging.error(str(e))


async def start_job_scheduler():
    logging.info("Starting Job Scheduler")
    task1: asyncio.Task = asyncio.create_task(loop_async_with_session(start_waiting_jobs))
    task2 = asyncio.create_task(loop_async_with_session(kill_errored_jobs))
    task3 = asyncio.create_task(loop_async_with_session(kube_cleanup_finished_jobs))
    await asyncio.gather(task1, task2, task3)


async def start_background_tasks(app):
    app['scheduler'] = asyncio.create_task(start_job_scheduler())


async def cleanup_background_tasks(app):
    app['scheduler'].cancel()
    await app['scheduler']


def main():
    app = web.Application()
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    app.add_routes(routes)
    port = PORT
    if port is None:
        port = 4000
    web.run_app(app, port=port)


if __name__ == "__main__":
    main()
