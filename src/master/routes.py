from .resources import DatasetLoadResource, DatasetListResource, \
    DatasetResource, Results, JobResource, JobListResource, \
    ExperimentResource, ExperimentListResource


def set_up_routes(api):
    api.add_resource(DatasetLoadResource, '/dataset/<int:dataset_id>/load')
    api.add_resource(DatasetResource, '/dataset/<int:dataset_id>')
    api.add_resource(DatasetListResource, '/datasets')
    api.add_resource(ExperimentResource, '/experiment/<int:experiment_id>')
    api.add_resource(ExperimentListResource, '/experiments')
    api.add_resource(JobResource, '/job/<int:job_id>')
    api.add_resource(JobListResource, '/jobs')

    api.add_resource(Results, '/results')
