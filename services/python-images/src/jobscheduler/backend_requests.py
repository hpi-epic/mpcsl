from aiohttp import ClientSession
import logging
from src.master.config import API_HOST


async def post_job_change(job_id, error_code):
    url = "http://" + API_HOST + "/api/job/" + str(job_id)
    logging.info("Emit change to %s", url)
    async with ClientSession() as session:
        async with session.post(url, json={'error_code': error_code}) as resp:
            resp.raise_for_status()
            logging.info(await resp.text())
