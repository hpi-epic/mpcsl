import asyncio
import logging

from aiohttp import web, ClientSession

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.master.config import DAEMON_CYCLE_TIME, SQLALCHEMY_DATABASE_URI, API_HOST, PORT
from src.models import Job, JobStatus, Experiment
from src.jobscheduler.kubernetes_helper import create_job, kube_cleanup_finished_jobs, get_node_list
from src.jobscheduler.kubernetes_helper import check_running_job, get_pod_log, delete_job_and_pods


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


async def post_job_change(job_id, status):
    url = "http://" + API_HOST + "/api/job/" + str(job_id)
    logging.info("Emit change to %s", url)
    async with ClientSession() as session:
        async with session.post(url, json={'status': status}) as resp:
            resp.raise_for_status()
            logging.info(await resp.text())


async def start_waiting_jobs(session):
    jobs = session.query(Job).filter(Job.status == JobStatus.waiting)
    for job in jobs:
        try:
            experiment = session.query(Experiment).get(job.experiment_id)
            k8s_job_name = await create_job(job, experiment)
            if isinstance(k8s_job_name, str):
                job.container_id = k8s_job_name
                job.status = JobStatus.running
                asyncio.create_task(post_job_change(job.id, job.status))
                session.commit()
        except Exception as e:
            logging.error(str(e))
            job.status = JobStatus.error
            asyncio.create_task(post_job_change(job.id, job.status))
            session.commit()


async def kill_errored_jobs(session):
    jobs = session.query(Job).filter(Job.status == JobStatus.running)
    for job in jobs:
        try:
            crashed = await check_running_job(job)
            if crashed:
                job: Job
                job.status = JobStatus.error
                asyncio.create_task(post_job_change(job.id, job.status))
                job.log = " -- EMPTY LOGS -- "  # TODO: Better error logs
                session.commit()
        except Exception as e:
            logging.error(str(e))


async def start_job_scheduler():
    logging.info("Starting Job Scheduler")
    while True:
        try:
            await asyncio.sleep(DAEMON_CYCLE_TIME)
            await asyncio.gather(start_waiting_jobs(Session()), kill_errored_jobs(Session()))
            kube_cleanup_finished_jobs(Session())
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
    app.add_routes(routes)
    port = PORT
    if port is None:
        port = 4000
    web.run_app(app, port=port)


if __name__ == "__main__":
    main()
