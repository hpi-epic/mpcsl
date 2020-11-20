
from src.db import db
from src.models.base import BaseSchema
from src.models.job import Job, JobSchema


class DatasetGenerationJob(Job):
    __tablename__ = 'dataset_generation_job'
    id = db.Column(db.Integer, db.ForeignKey('job.id'), primary_key=True)

    dataset_id = db.Column(
        db.Integer,
        db.ForeignKey('dataset.id'),
        nullable=False
    )
    experiment = db.relationship(
        'Dataset',
        backref=db.backref('job', cascade="all, delete-orphan", uselist=False)
    )

    nodes = db.Column(db.Integer)
    samples = db.Column(db.Integer)
    edgeProbability = db.Column(db.Float)
    edgeValueLowerBound = db.Column(db.Float)
    edgeValueUpperBound = db.Column(db.Float)

    __mapper_args__ = {
        'polymorphic_identity': 'dataset_generation_job',
    }


class DatasetGenerationJobSchema(JobSchema):
    class Meta(BaseSchema.Meta):
        model = DatasetGenerationJob
gi