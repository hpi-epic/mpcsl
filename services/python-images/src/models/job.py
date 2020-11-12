import enum

from marshmallow import fields


from src.db import db
from src.models.base import BaseModel, BaseSchema


class JobStatus(str, enum.Enum):
    waiting = "waiting"
    running = "running"
    done = "done"
    error = "error"
    cancelled = "cancelled"
    hidden = "hidden"


class JobErrorCode(int, enum.Enum):
    UNSCHEDULABLE = -1
    IMAGE_NOT_FOUND = -2
    UNKNOWN = -127


class Job(BaseModel):
    start_time = db.Column(db.DateTime, nullable=False)
    container_id = db.Column(db.String)
    node_hostname = db.Column(db.String)
    status = db.Column(db.Enum(JobStatus), nullable=False)
    log = db.Column(db.String)
    error_code = db.Column(db.Enum(JobErrorCode))

    @property
    def result(self):
        if len(self.results) == 0:
            return None
        return sorted(self.results, key=lambda x: x.start_time)[-1]


class JobSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Job

    result = fields.Nested('ResultSchema')
