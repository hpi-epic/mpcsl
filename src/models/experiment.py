from src.db import db
from src.models.base import BaseModel, BaseSchema


class Experiment(BaseModel):
    alpha = db.Column(db.Float)

    independence_test = db.Column(db.String)

    dataset_id = db.Column(db.Integer)
    # TODO: Dataset relationship

    cores = db.Column(db.Integer)


class ExperimentSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Experiment
