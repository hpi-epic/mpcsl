from marshmallow import fields, Schema
from marshmallow.validate import OneOf

from src.db import db
from src.models.base import BaseModel, BaseSchema
from src.models.swagger import SwaggerMixin

INDEPENDENCE_TESTS = ["gaussCI", "disCI", "binCI"]


class Experiment(BaseModel):
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    dataset = db.relationship('Dataset')

    name = db.Column(db.String)
    parameters = db.Column(db.JSON)


class ExperimentParameterSchema(Schema, SwaggerMixin):
    alpha = fields.Float(required=True)
    independence_test = fields.String(required=True, validate=OneOf(INDEPENDENCE_TESTS))
    cores = fields.Integer(required=True)


class ExperimentSchema(BaseSchema):
    parameters = fields.Nested(ExperimentParameterSchema)

    class Meta(BaseSchema.Meta):
        model = Experiment
