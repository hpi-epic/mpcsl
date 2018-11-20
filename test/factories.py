import factory
from factory.alchemy import SQLAlchemyModelFactory


from src.models import *
from src.db import db


class BaseFactory(SQLAlchemyModelFactory):
    class Meta:
        model = BaseModel
        abstract = False
        sqlalchemy_session = db.session
        strategy = factory.CREATE_STRATEGY
        force_flush = True


class DatasetFactory(BaseFactory):
    class Meta:
        model = Dataset
        sqlalchemy_session = db.session

    name = factory.Faker('word')
    load_query = factory.Faker('file_path')
