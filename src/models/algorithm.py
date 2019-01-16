import enum

from marshmallow import fields
from sqlalchemy.ext.mutable import MutableDict

from src.db import db
from src.models.base import BaseModel, BaseSchema


class BackendType(str, enum.Enum):
    python = "python"
    cpp = "cpp"
    R = "R"


class Algorithm(BaseModel):
    name = db.Column(db.String, unique=True)
    script_filename = db.Column(db.String)
    backend = db.Column(db.Enum(BackendType))
    description = db.Column(db.String)
    valid_parameters = db.Column(MutableDict.as_mutable(db.JSON))


class AlgorithmSchema(BaseSchema):
    valid_parameters = fields.Dict()

    class Meta(BaseSchema.Meta):
        model = Algorithm
