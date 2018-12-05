import factory
import random
from factory.alchemy import SQLAlchemyModelFactory


from src.models import BaseModel, Dataset, Experiment, Job
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
