from src.db import db
from src.models.base import BaseModel, BaseSchema


class Experiment(BaseModel):
    pass


class ExperimentSchema(BaseSchema):
    class Meta:
        model = Experiment
