from flask import request


def load_data(schema, location='json', args=None, kwargs=None):
    vals = getattr(request, location, None)
    return schema().load(vals, *args, **kwargs)


def marshal(schema, object, *args, **kwargs):
    return schema().dump(object, *args, **kwargs).data

