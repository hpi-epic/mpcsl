
from src.db import db
from src.models.base import BaseSchema
from src.models.job import Job, JobSchema


class DatasetGenerationJob(Job):
    __tablename__ = 'dataset_generation_job'
    id = db.Column(db.Integer, db.ForeignKey('job.id'), primary_key=True)

    dataset_id = db.Column(
        db.Integer,
        db.ForeignKey('dataset.id'),
        nullable=True
    )
    dataset = db.relationship(
        'Dataset',
        backref=db.backref('job', cascade="all, delete-orphan", uselist=False)
    )

    parameters = db.Column(db.JSON, nullable=False)
    generator_type = db.Column(db.String, nullable=False)
    datasetName = db.Column(db.String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'dataset_generation_job',
    }


class DatasetGenerationJobSchema(JobSchema):
    class Meta(BaseSchema.Meta):
        model = DatasetGenerationJob
