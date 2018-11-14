from .resources.hello import HelloWorld


def set_up_routes(api):
    api.add_resource(HelloWorld, '/hello')
