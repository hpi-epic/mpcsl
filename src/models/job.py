import enum

from src.db import db
from marshmallow import fields
from src.models.base import BaseModel, BaseSchema


class JobStatus(str, enum.Enum):
    running = "running"
    done = "done"
    error = "error"
    cancelled = "cancelled"


class Job(BaseModel):
    experiment_id = db.Column(
        db.Integer,
        db.ForeignKey('experiment.id'),
        nullable=False
    )
    experiment = db.relationship('Experiment', backref=db.backref('jobs', cascade="all, delete-orphan"))
    start_time = db.Column(db.DateTime, nullable=False)
    pid = db.Column(db.Integer)
    status = db.Column(db.Enum(JobStatus), nullable=False)

    @property
    def result(self):
        if len(self.results) == 0:
            return None
        return sorted(self.results, key=lambda x: x.start_time)[-1]


class JobSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Job

    result = fields.Nested('ResultSchema')
