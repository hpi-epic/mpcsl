from .resources import DatasetLoadResource, DatasetListResource, \
    DatasetResource, JobResource, JobListResource, \
    ExperimentResource, ExperimentListResource, ResultListResource, \
    JobLogsResource, JobResultResource, ExperimentJobListResource, ResultResource, \
    AlgorithmResource, AlgorithmListResource, \
    ResultNodeListResource, ResultEdgeListResource, ResultSepsetListResource, GraphExportResource, \
    NodeResource, EdgeResource, SepsetResource, NodeContextResource, MarginalDistributionResource, \
    DatasetAvailableSourcesResource, \
    ConditionalDistributionResource, ExecutorResource


def base_url(url):
    return '/api' + url


def set_up_routes(api):
    api.add_resource(AlgorithmListResource, base_url('/algorithms'))
    api.add_resource(AlgorithmResource, base_url('/algorithm/<int:algorithm_id>'))
    api.add_resource(DatasetLoadResource, base_url('/dataset/<int:dataset_id>/load'))
    api.add_resource(DatasetResource, base_url('/dataset/<int:dataset_id>'))
    api.add_resource(DatasetListResource, base_url('/datasets'))
    api.add_resource(DatasetAvailableSourcesResource, base_url('/datasources'))
    api.add_resource(ExperimentResource, base_url('/experiment/<int:experiment_id>'))
    api.add_resource(ExperimentJobListResource, base_url('/experiment/<int:experiment_id>/jobs'))
    api.add_resource(ExecutorResource, base_url('/experiment/<int:experiment_id>/start'))
    api.add_resource(ExperimentListResource, base_url('/experiments'))
    api.add_resource(JobResource, base_url('/job/<int:job_id>'))
    api.add_resource(JobLogsResource, base_url('/job/<int:job_id>/logs'))
    api.add_resource(JobResultResource, base_url('/job/<int:job_id>/result'))
    api.add_resource(JobListResource, base_url('/jobs'))
    api.add_resource(ResultListResource, base_url('/results'))
    api.add_resource(ResultResource, base_url('/result/<int:result_id>'))
    api.add_resource(ResultNodeListResource, base_url('/result/<int:result_id>/nodes'))
    api.add_resource(ResultEdgeListResource, base_url('/result/<int:result_id>/edges'))
    api.add_resource(ResultSepsetListResource, base_url('/result/<int:result_id>/sepsets'))
    api.add_resource(GraphExportResource, base_url('/result/<int:result_id>/export'))
    api.add_resource(NodeResource, base_url('/node/<int:node_id>'))
    api.add_resource(NodeContextResource, base_url('/node/<int:node_id>/context'))
    api.add_resource(MarginalDistributionResource, base_url('/node/<int:node_id>/marginal'))
    api.add_resource(ConditionalDistributionResource, base_url('/node/<int:node_id>/conditional'))
    api.add_resource(EdgeResource, base_url('/edge/<int:edge_id>'))
    api.add_resource(SepsetResource, base_url('/sepset/<int:sepset_id>'))
