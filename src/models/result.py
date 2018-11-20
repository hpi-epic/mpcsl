from src.db import db
from src.models.base import BaseModel, BaseSchema


class Result(BaseModel):
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'), nullable=False)
    experiment = db.relationship('Experiment')

    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    meta_results = db.Column(db.String)


class ResultSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Result