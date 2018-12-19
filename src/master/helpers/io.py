from flask import request


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
        raise InvalidInputData(payload=errors)
    return data


def marshal(schema, object, *args, **kwargs):
    return schema().dump(object, *args, **kwargs).data
