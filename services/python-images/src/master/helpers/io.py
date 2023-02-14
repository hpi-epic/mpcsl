from flask import request
from werkzeug.exceptions import BadRequest


class InvalidInputData(BadRequest):
    def __init__(self, message='Invalid input data.', payload=None):
        self.payload = payload
        BadRequest.__init__(self, description=payload or message)


def load_data(schema, location='json', *args, **kwargs):
    vals = getattr(request, location, None)
    try:
        data = schema().load(vals, *args, **kwargs)
    except Exception as err:
        raise InvalidInputData(payload=err)
    return data


def marshal(schema, object, *args, **kwargs):
    return schema().dump(object, *args, **kwargs)
