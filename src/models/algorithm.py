import enum

from marshmallow import fields
from sqlalchemy.ext.mutable import MutableDict

from src.db import db
from src.models.base import BaseModel, BaseSchema
from src.models.swagger import SwaggerMixin


class BackendType(str, enum.Enum):
    python = "python"
    cpp = "cpp"
    R = "R"


class Algorithm(BaseModel):
    name = db.Column(db.String, unique=True)
    script_filename = db.Column(db.String)
    backend = db.Column(db.Enum(BackendType))
    description = db.Column(db.String)

    @property
    def valid_parameters(self):
        if len(self.valid_parameters) == 0:
            return None
        return sorted(self.valid_parameters, key=lambda x: x.name)


class AlgorithmSchema(BaseSchema, SwaggerMixin):

    class Meta(BaseSchema.Meta):
        model = Algorithm

    valid_parameters = fields.Nested('ParameterSchema', many=True)
