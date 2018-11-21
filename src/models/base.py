from src.db import db
from marshmallow_sqlalchemy import ModelSchema
from marshmallow import fields


class BaseModel(db.Model):
    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)

    def update(self, data):
        for key, value in data.items():
            setattr(self, key, value)


class BaseSchema(ModelSchema):
    id = fields.Integer(dump_only=True)

    class Meta:
        sqla_session = db.session
        include_fk = True

    def make_instance(self, data):
        # Overridden to disable automatic loading by SQLAlchemy-Marshmallow.
        return data
