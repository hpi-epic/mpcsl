from src.db import db
from marshmallow_sqlalchemy import ModelSchema
from marshmallow import fields


class BaseModel(db.Model):
    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseSchema(ModelSchema):
    id = fields.Integer(dump_only=True)
