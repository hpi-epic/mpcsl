import enum

from src.db import db
from src.models.base import BaseModel, BaseSchema


class EdgeAnnotation(str, enum.Enum):
    valid = "valid"
    invalid = "invalid"


class EdgeInformation(BaseModel):
    edge_id = db.Column(
        db.Integer,
        db.ForeignKey('edge.id'),
        nullable=False
    )
    edge = db.relationship('Edge',
                           backref=db.backref('edge_information', cascade="all, delete-orphan"))

    annotation = db.Column(db.Enum(EdgeAnnotation), nullable=False)


class EdgeInformationSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = EdgeInformation
