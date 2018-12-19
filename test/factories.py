import factory
import random
from factory.alchemy import SQLAlchemyModelFactory


from src.models import BaseModel, Dataset, Experiment, Job, Result, Node, Edge, Sepset, JobStatus
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


class ExperimentFactory(BaseFactory):
    class Meta:
        model = Experiment
        sqlalchemy_session = db.session

    dataset = factory.SubFactory(DatasetFactory)
    parameters = factory.LazyAttribute(lambda o: {
        'alpha': round(random.random(), 2),
        'independence_test': 'gaussCI',
        'cores': 1
    })


class JobFactory(BaseFactory):
    class Meta:
        model = Job
        sqlalchemy_session = db.session

    experiment = factory.SubFactory(ExperimentFactory)
    start_time = factory.Faker('date_time')
    pid = factory.Faker('pyint')
    status = JobStatus.running


class ResultFactory(BaseFactory):
    class Meta:
        model = Result
        sqlalchemy_session = db.session

    job = factory.SubFactory(JobFactory)
    start_time = factory.Faker('date_time')
    end_time = factory.Faker('date_time')


class NodeFactory(BaseFactory):
    class Meta:
        model = Node
        sqlalchemy_session = db.session

    result = factory.SubFactory(ResultFactory)
    name = factory.Faker('word')


class EdgeFactory(BaseFactory):
    class Meta:
        model = Edge
        sqlalchemy_session = db.session

    result = factory.SubFactory(ResultFactory)
    from_node = factory.SubFactory(NodeFactory)
    to_node = factory.SubFactory(NodeFactory)


class SepsetFactory(BaseFactory):
    class Meta:
        model = Sepset
        sqlalchemy_session = db.session

    result = factory.SubFactory(ResultFactory)
    from_node = factory.SubFactory(NodeFactory)
    to_node = factory.SubFactory(NodeFactory)

    level = random.randint(1, 5)
    statistic = random.random()
    node_names = factory.List([factory.Faker('word') for _ in range(random.randint(1, 5))])
