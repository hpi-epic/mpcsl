import os
from flask import request
from src.master.config import LOGS_DIRECTORY


class InvalidInputData(Exception):
    status_code = 400

    def __init__(self, message='Invalid input data.', status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


def load_data(schema, location='json', *args, **kwargs):
    vals = getattr(request, location, None)
    data, errors = schema().load(vals, *args, **kwargs)
    if len(errors) > 0:
        exc = InvalidInputData(payload=errors)
        print(exc.to_dict())
        raise exc
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
