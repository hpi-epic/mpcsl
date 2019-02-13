import enum

from src.db import db
from src.models.base import BaseModel, BaseSchema


class EdgeAnnotation(str, enum.Enum):
    valid = "valid"
    invalid = "invalid"


class EdgeInformation(BaseModel):
    experiment_id = db.Column(
        db.Integer,
        db.ForeignKey('experiment.id'),
        nullable=False
    )

    experiment = db.relationship('Experiment',
                                 backref=db.backref('jobs',
                                                    cascade="all, delete-orphan"))

    annotation = db.Column(db.Enum(EdgeAnnotation), nullable=False)

    from_node_name = db.Column(db.String)

    to_node_name = db.Column(db.String)


class EdgeInformationSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = EdgeInformation
