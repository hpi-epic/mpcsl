from .resources.dataset import DataSetResource, DataSetListResource


def set_up_routes(api):
    api.add_resource(DataSetResource, '/dataset/<data_set_id:int>')
    api.add_resource(DataSetListResource, '/datasets')
