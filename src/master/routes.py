from .resources import Datasets, HelloWorld, Results


def set_up_routes(api):
    api.add_resource(HelloWorld, '/hello')
    api.add_resource(Datasets, '/datasets/<int:dataset_id>')
    api.add_resource(Results, '/results')
