import asyncio
import docker
import logging
import aiohttp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.master.config import DAEMON_CYCLE_TIME, SQLALCHEMY_DATABASE_URI, API_HOST
from src.models import Job, JobStatus
from src.master.helpers.docker import get_client

logging.basicConfig(level=logging.DEBUG)


async def post_job_change(session, job_id, status):
    url = "http://" + API_HOST + "/api/job/" + str(job_id)
    logging.info("Emit change to %s", url)
    async with session.post(url, json={'status': status}) as resp:
        resp.raise_for_status()
        logging.info(await resp.text())


async def start_job_scheduler():
    logging.info("Starting Job Scheduler")
    client = get_client()
    engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
    Session = sessionmaker(bind=engine)
    http_session = aiohttp.ClientSession()
    while True:
        tasks = []
        await asyncio.sleep(DAEMON_CYCLE_TIME)
        session = Session()
        jobs = session.query(Job).filter(Job.status == JobStatus.running)

        for job in jobs:
            try:
                container = client.containers.get(job.container_id)
                if container.status == "exited":
                    logging.warning("Container with id %s exited", job.container_id)
                    job.status = JobStatus.error
                    tasks.append(asyncio.create_task(post_job_change(http_session, job.id, job.status)))
            except docker.errors.NotFound:
                logging.warning("Container with id %s not found", job.container_id)
                job.status = JobStatus.error
                tasks.append(asyncio.create_task(post_job_change(http_session, job.id, job.status)))
            session.commit()
        session.close()
        if len(tasks) != 0:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(start_job_scheduler())
