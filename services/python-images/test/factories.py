import factory
import random
from factory.alchemy import SQLAlchemyModelFactory

from src.models import Algorithm, BaseModel, Dataset, Edge, EdgeAnnotation, EdgeInformation, \
    Experiment, ExperimentJob, DatasetGenerationJob, Job, JobStatus, Node, Result, Sepset
from src.db import db


class BaseFactory(SQLAlchemyModelFactory):
    class Meta:
        model = BaseModel
        abstract = False
        sqlalchemy_session = db.session
        strategy = factory.CREATE_STRATEGY
        sqlalchemy_session_persistence = 'commit'


class DatasetFactory(BaseFactory):
    class Meta:
        model = Dataset
        sqlalchemy_session = db.session

    name = factory.Faker('word')
    load_query = factory.Faker('file_path')
    description = factory.Faker('text')
    content_hash = factory.Faker('md5')
    data_source = 'postgres'


class AlgorithmFactory(BaseFactory):
    class Meta:
        model = Algorithm
        sqlalchemy_session = db.session

    package = factory.Faker('word')
    function = factory.Faker('word')
    script_filename = 'pcalg.r'
    description = ''
    docker_image = 'mpci_execution_r'
    valid_parameters = factory.LazyAttribute(lambda o: {
        'alpha': {'type': 'float', 'required': True, 'minimum': 0.0, 'maximum': 1.0},
        'independence_test': {'type': 'enum', 'required': True, 'values': ['gaussCI', 'disCI', 'binCI']},
        'cores': {'type': 'int', "minimum": 1},
        'verbose': {'type': 'int', 'minimum': 0, 'maximum': 1, 'default': 0},
        'subset_size': {'type': 'int', 'minimum': -1, 'default': -1}
    })


class ExperimentFactory(BaseFactory):
    class Meta:
        model = Experiment
        sqlalchemy_session = db.session

    name = factory.Faker('word')
    description = factory.Faker('text')

    dataset = factory.SubFactory(DatasetFactory)
    algorithm = factory.SubFactory(AlgorithmFactory)
    parameters = factory.LazyAttribute(lambda o: {
        'alpha': round(random.random(), 2),
        'independence_test': 'gaussCI',
        'cores': 1,
        'verbose': 0,
        'subset_size': -1
    })


class JobFactory(BaseFactory):
    class Meta:
        model = Job
        sqlalchemy_session = db.session

    start_time = factory.Faker('date_time')
    container_id = factory.Faker('md5')
    status = JobStatus.running


class ExperimentJobFactory(JobFactory):
    class Meta:
        model = ExperimentJob
        sqlalchemy_session = db.session

    experiment = factory.SubFactory(ExperimentFactory)


class DatasetGenerationJobFactory(JobFactory):
    class Meta:
        model = DatasetGenerationJob
        sqlalchemy_session = db.session

    parameters = factory.LazyAttribute(lambda o: {
        "num_nodes": 10,
        "num_samples": 10000
    })
    generator_type = "mpci"


class ResultFactory(BaseFactory):
    class Meta:
        model = Result
        sqlalchemy_session = db.session

    job = factory.SubFactory(ExperimentJobFactory)
    start_time = factory.Faker('date_time')
    end_time = factory.Faker('date_time')


class NodeFactory(BaseFactory):
    class Meta:
        model = Node
        sqlalchemy_session = db.session

    dataset = factory.SubFactory(DatasetFactory)
    name = factory.Faker('word')


class EdgeFactory(BaseFactory):
    class Meta:
        model = Edge
        sqlalchemy_session = db.session

    result = factory.SubFactory(ResultFactory)
    from_node = factory.SubFactory(NodeFactory)
    to_node = factory.SubFactory(NodeFactory)
    weight = random.random()


class EdgeInformationFactory(BaseFactory):
    class Meta:
        model = EdgeInformation
        sqlalchemy_session = db.session

    result = factory.SubFactory(ResultFactory)
    from_node = factory.SubFactory(NodeFactory)
    to_node = factory.SubFactory(NodeFactory)

    annotation = random.choice(list(EdgeAnnotation))


class SepsetFactory(BaseFactory):
    class Meta:
        model = Sepset
        sqlalchemy_session = db.session

    result = factory.SubFactory(ResultFactory)
    from_node = factory.SubFactory(NodeFactory)
    to_node = factory.SubFactory(NodeFactory)

    level = random.randint(1, 5)
    statistic = random.random()
