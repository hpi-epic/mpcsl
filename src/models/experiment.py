import enum

from src.db import db
from src.models.base import BaseModel, BaseSchema


class IndependenceTest(str, enum.Enum):
    gaussCI = "gaussCI"
    binCI = "binCI"
    disCI = "disCI"


class Experiment(BaseModel):
    alpha = db.Column(db.Float)

    independence_test = db.Column(db.Enum(IndependenceTest))

    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    dataset = db.relationship('Dataset')

    cores = db.Column(db.Integer)


class ExperimentSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Experiment
