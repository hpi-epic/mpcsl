import factory
from factory.alchemy import SQLAlchemyModelFactory


from src.models import BaseModel, Dataset, Job
from src.db import db


class BaseFactory(SQLAlchemyModelFactory):
    class Meta:
        model = BaseModel
        abstract = False
        sqlalchemy_session = db.session
        strategy = factory.CREATE_STRATEGY
        sqlalchemy_session_persistence = 'flush'


class DatasetFactory(BaseFactory):
    class Meta:
        model = Dataset
        sqlalchemy_session = db.session

    name = factory.Faker('word')
    load_query = factory.Faker('file_path')


class JobFactory(BaseFactory):
    class Meta:
        model = Job
        sqlalchemy_session = db.session

    experiment = None  # factory.SubFactory(ExperimentFactory)
    start_time = factory.Faker('date_time')
    pid = factory.Faker('pyint')
