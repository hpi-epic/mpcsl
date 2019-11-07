import enum
from sqlalchemy import event

from src.db import db
from marshmallow import fields
from src.models.base import BaseModel, BaseSchema
from src.master.helpers.socketio_events import job_status_change


class JobStatus(str, enum.Enum):
    running = "running"
    done = "done"
    error = "error"
    cancelled = "cancelled"
    hidden = "hidden"


class Job(BaseModel):
    experiment_id = db.Column(
        db.Integer,
        db.ForeignKey('experiment.id'),
        nullable=False
    )
    experiment = db.relationship('Experiment', backref=db.backref('jobs', cascade="all, delete-orphan"))
    start_time = db.Column(db.DateTime, nullable=False)
    container_id = db.Column(db.String)
    status = db.Column(db.Enum(JobStatus), nullable=False)

    @property
    def result(self):
        if len(self.results) == 0:
            return None
        return sorted(self.results, key=lambda x: x.start_time)[-1]


@event.listens_for(Job, 'after_update')
def emitJobChange(mapper, connection, target):
    job_status_change(target.id, target.status)


class JobSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Job

    result = fields.Nested('ResultSchema')
