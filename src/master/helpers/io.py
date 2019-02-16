import os
from flask import request
from werkzeug.exceptions import BadRequest

from src.master.config import LOGS_DIRECTORY


class InvalidInputData(BadRequest):
    def __init__(self, message='Invalid input data.', payload=None):
        self.payload = payload
        BadRequest.__init__(self, description=payload or message)


def load_data(schema, location='json', *args, **kwargs):
    vals = getattr(request, location, None)
    data, errors = schema().load(vals, *args, **kwargs)
    if len(errors) > 0:
        raise InvalidInputData(payload=errors)
    return data


def marshal(schema, object, *args, **kwargs):
    return schema().dump(object, *args, **kwargs).data


def silent_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def get_logfile_name(job_id):
    return f'{LOGS_DIRECTORY}/job_{job_id}.log'


def get_r_logfile_name(job_id):
    return f'{LOGS_DIRECTORY}/job_{job_id}_error.RData'


def remove_logs(job_id):
    logfile = get_logfile_name(job_id)
    request_file = get_r_logfile_name(job_id)

    silent_remove(logfile)
    silent_remove(request_file)
