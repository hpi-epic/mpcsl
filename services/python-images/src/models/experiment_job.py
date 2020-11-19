
from src.db import db
from src.models.base import BaseModel, BaseSchema
from src.models.job import Job, JobSchema
from src.models.experiment import Experiment

class ExperimentJob(Job):
    __tablename__ = 'experiment_job'
    id = db.Column(db.Integer, db.ForeignKey('job.id'), primary_key=True)

    experiment_id = db.Column(
        db.Integer,
        db.ForeignKey('experiment.id'),
        nullable=False
    )
    experiment = db.relationship('Experiment', backref=db.backref('jobs', cascade="all, delete-orphan"))
    
    parallel = db.Column(db.Boolean)
    enforce_cpus = db.Column(db.Boolean, default=True)
    gpus = db.Column(db.Integer)

    __mapper_args__ = {
        'polymorphic_identity':'experiment_job',
    }


class ExperimentJobSchema(JobSchema):
    class Meta(BaseSchema.Meta):
        model = ExperimentJob