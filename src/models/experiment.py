from marshmallow import fields, Schema
from marshmallow.validate import OneOf, Length
from marshmallow_sqlalchemy import field_for
from sqlalchemy.ext.mutable import MutableDict

from src.db import db
from src.models.base import BaseModel, BaseSchema
from src.models.swagger import SwaggerMixin

INDEPENDENCE_TESTS = ["gaussCI", "disCI", "binCI"]


class Experiment(BaseModel):
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    dataset = db.relationship('Dataset')

    name = db.Column(db.String)
    algorithm_id = db.Column(db.Integer, db.ForeignKey('algorithm.id'))
    algorithm = db.relationship('Algorithm')

    description = db.Column(db.String)
    parameters = db.Column(MutableDict.as_mutable(db.JSON))

    @property
    def last_job(self):
        if len(self.jobs) == 0:
            return None
        return sorted(self.jobs, key=lambda x: x.start_time)[-1]


class ExperimentParameterSchema(Schema, SwaggerMixin):
    alpha = fields.Float(required=True)
    independence_test = fields.String(required=True, validate=OneOf(INDEPENDENCE_TESTS))
    cores = fields.Integer(required=True)


class ExperimentSchema(BaseSchema):
    name = field_for(Experiment, 'name', required=True, validate=Length(min=1))
    description = field_for(Experiment, 'description', required=False, allow_none=True, default='')
    algorithm = fields.Nested('AlgorithmSchema')
    parameters = fields.Nested(ExperimentParameterSchema)
    last_job = fields.Nested('JobSchema', dump_only=True)

    class Meta(BaseSchema.Meta):
        model = Experiment
