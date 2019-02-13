from src.db import db
from src.models.base import BaseModel, BaseSchema
from marshmallow_sqlalchemy import field_for
from marshmallow.validate import Length, OneOf
from marshmallow import fields


class Parameter(BaseModel):
    algorithm_id = db.Column(db.Integer, db.ForeignKey('algorithm.id'), nullable=False)
    algorithm = db.relationship('Algorithm', backref=db.backref('parameters', cascade="all, delete-orphan"))

    name = db.Column(db.String, nullable=False)
    type = db.Column(db.String, nullable=False)
    required = db.Column(db.Boolean, nullable=False)
    values = db.Column(db.ARRAY(db.String))
    default = db.Column(db.String)
    min = db.Column(db.String)
    max = db.Column(db.String)


TYPE_MAP = {
    'str': lambda required: fields.String(required=required, validate=Length(min=0)),
    'int': lambda required: fields.Integer(required=required),
    'float': lambda required: fields.Float(required=required),
    'bool': lambda required: fields.Boolean(required=required),
}


class ParameterSchema(BaseSchema):
    name = field_for(Parameter, 'name', required=True, validate=Length(min=1))
    type = field_for(Parameter, 'type', required=True, validate=OneOf(list(TYPE_MAP.keys())))
    required = field_for(Parameter, 'required', default=False, validate=OneOf(['true', 'false']))
    values = field_for(Parameter, 'values', required=False)
    default = field_for(Parameter, 'default', required=False)
    min = field_for(Parameter, 'min', required=False)
    max = field_for(Parameter, 'max', required=False)

    class Meta(BaseSchema.Meta):
        model = Parameter
