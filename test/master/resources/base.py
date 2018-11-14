import json

from test.master.base import BaseTest


class BaseResourceTest(BaseTest):

    def get(self, *args, **kwargs):
        return self.call(self.test_client.get, args, kwargs)

    def put(self, *args, **kwargs):
        return self.call(self.test_client.put, args, kwargs)

    def delete(self, *args, **kwargs):
        return self.call(self.test_client.delete, args, kwargs)

    def post(self, *args, **kwargs):
        return self.call(self.test_client.post, args, kwargs)

    def call(self, method, args, kwargs):
        if 'json' in kwargs:
            kwargs['data'] = json.dumps(kwargs['json'])

        result = method(*args, **kwargs)

        return json.loads(result.data)
