import factory
import random
from factory.alchemy import SQLAlchemyModelFactory


from src.models import BaseModel, Dataset, Experiment
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


class ExperimentFactory(BaseFactory):
    class Meta:
        model = Experiment
        sqlalchemy_session = db.session

    alpha = factory.LazyAttribute(lambda o: random.randint(0, 100)/100)
    cores = factory.LazyAttribute(lambda o: random.randint(0, 4))
    dataset = factory.SubFactory(DatasetFactory)
