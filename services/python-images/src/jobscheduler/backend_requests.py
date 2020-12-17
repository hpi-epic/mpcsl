from aiohttp import ClientSession
import logging
from src.master.config import API_HOST
from src.models import JobErrorCode


async def post_job_change(job_id: int, error_code: JobErrorCode):
    url = "http://" + API_HOST + "/api/job/" + str(job_id)
    logging.info(f"Emit change with error_code {error_code} to {job_id}")
    async with ClientSession() as session:
        async with session.post(url, json={'error_code': error_code}) as resp:
            resp.raise_for_status()
            logging.info(await resp.text())
