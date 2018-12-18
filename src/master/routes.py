from .resources import DatasetLoadResource, DatasetListResource, \
    DatasetResource, JobResource, JobListResource, \
    ExperimentResource, ExperimentListResource, ResultListResource, \
    JobResultResource, ExperimentJobListResource, ResultResource
from src.master.executor.executor import ExecutorResource


def set_up_routes(api):
    api.add_resource(DatasetLoadResource, '/dataset/<int:dataset_id>/load')
    api.add_resource(DatasetResource, '/dataset/<int:dataset_id>')
    api.add_resource(DatasetListResource, '/datasets')
    api.add_resource(ExperimentResource, '/experiment/<int:experiment_id>')
    api.add_resource(ExperimentJobListResource, '/experiment/<int:experiment_id>/jobs')
    api.add_resource(ExecutorResource, '/experiment/<int:experiment_id>/start')
    api.add_resource(ExperimentListResource, '/experiments')
    api.add_resource(JobResource, '/job/<int:job_id>')
    api.add_resource(JobResultResource, '/job/<int:job_id>/result')
    api.add_resource(JobListResource, '/jobs')
    api.add_resource(ResultListResource, '/results')
    api.add_resource(ResultResource, '/result/')
