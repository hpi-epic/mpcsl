
from src.db import db
from src.models.base import BaseModel

class ExperimentJob(BaseModel):
    job_id = db.Column(
        db.Integer,
        db.ForeignKey('job.id')
    )
    job = db.relationship('Job', backref=db.backref('experiment_job', cascade="all, delete-orphan"))
    experiment_id = db.Column(
        db.Integer,
        db.ForeignKey('experiment.id'),
        nullable=False
    )
    experiment = db.relationship('Experiment', backref=db.backref('jobs', cascade="all, delete-orphan"))
    
    parallel = db.Column(db.Boolean)
    enforce_cpus = db.Column(db.Boolean, default=True)
    gpus = db.Column(db.Integer)


class ExperimentJobSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = ExperimentJob