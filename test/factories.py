import factory
import random
from factory.alchemy import SQLAlchemyModelFactory

from src.models import BaseModel, Dataset, Experiment, Job, Result, Node, Edge, Sepset, Algorithm
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


class AlgorithmFactory(BaseFactory):
    class Meta:
        model = Algorithm
        sqlalchemy_session = db.session

    name = factory.Faker('word')
    script_filename = 'pcalg.r'
    description = ''
    backend = 'R'
    valid_parameters = dict()


class ExperimentFactory(BaseFactory):
    class Meta:
        model = Experiment
        sqlalchemy_session = db.session

    dataset = factory.SubFactory(DatasetFactory)
    algorithm = factory.SubFactory(AlgorithmFactory)
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

    name = factory.Faker('word')


class EdgeFactory(BaseFactory):
    class Meta:
        model = Edge
        sqlalchemy_session = db.session


class SepsetFactory(BaseFactory):
    class Meta:
        model = Sepset
        sqlalchemy_session = db.session

    level = random.randint(1, 5)
    statistic = random.random()
