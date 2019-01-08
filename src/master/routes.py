from .resources import DatasetLoadResource, DatasetListResource, \
    DatasetResource, JobResource, JobListResource, \
    ExperimentResource, ExperimentListResource, ResultListResource, \
    JobResultResource, ExperimentJobListResource, ResultResource, \
    AlgorithmResource, AlgorithmListResource
from src.master.executor.executor import ExecutorResource


def base_url(url):
    return '/api' + url


def set_up_routes(api):
    api.add_resource(AlgorithmListResource, base_url('/algorithms'))
    api.add_resource(AlgorithmResource, base_url('/algorithm/<int:algorithm_id>'))
    api.add_resource(DatasetLoadResource, base_url('/dataset/<int:dataset_id>/load'))
    api.add_resource(DatasetResource, base_url('/dataset/<int:dataset_id>'))
    api.add_resource(DatasetListResource, base_url('/datasets'))
    api.add_resource(ExperimentResource, base_url('/experiment/<int:experiment_id>'))
    api.add_resource(ExperimentJobListResource, base_url('/experiment/<int:experiment_id>/jobs'))
    api.add_resource(ExecutorResource, base_url('/experiment/<int:experiment_id>/start'))
    api.add_resource(ExperimentListResource, base_url('/experiments'))
    api.add_resource(JobResource, base_url('/job/<int:job_id>'))
    api.add_resource(JobResultResource, base_url('/job/<int:job_id>/result'))
    api.add_resource(JobListResource, base_url('/jobs'))
    api.add_resource(ResultListResource, base_url('/results'))
    api.add_resource(ResultResource, base_url('/result/<int:result_id>'))
