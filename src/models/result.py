from src.db import db
from src.models.base import BaseModel, BaseSchema


class Result(BaseModel):
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    job = db.relationship('Job', backref=db.backref('results'))

    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    meta_results = db.Column(db.JSON)


class ResultSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        exclude = ['edge_informations']
        model = Result
