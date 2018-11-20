from src.db import db
from src.models.base import BaseModel, BaseSchema


class Job(BaseModel):
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiment.id'), nullable=False)
    experiment = db.relationship('Experiment')

    start_time = db.Column(db.DateTime, nullable=False)
    pid = db.Column(db.Integer)


class JobSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Job
