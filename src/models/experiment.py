from marshmallow import fields, Schema
from marshmallow.validate import OneOf

from src.db import db
from src.models.base import BaseModel, BaseSchema

INDEPENDENCE_TESTS = ["gaussCI", "disCI"]


class Experiment(BaseModel):
    parameters = db.Column(db.JSON)

    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    dataset = db.relationship('Dataset')

    name = db.Column(db.String)


class ExperimentParameterSchema(Schema):
    alpha = fields.Float(required=True)
    independence_test = fields.String(required=True, validate=OneOf(INDEPENDENCE_TESTS))
    cores = fields.Integer(required=True)


class ExperimentSchema(BaseSchema):
    parameters = fields.Nested(ExperimentParameterSchema)

    class Meta(BaseSchema.Meta):
        model = Experiment
