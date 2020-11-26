import enum

from src.db import db
from src.models.base import BaseModel, BaseSchema


class EdgeAnnotation(str, enum.Enum):
    approved = "approved"
    declined = "declined"
    missing = "missing"


class EdgeInformation(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'), nullable=False)
    result = db.relationship('Result', backref=db.backref('edge_informations', cascade="all, delete-orphan"))

    from_node_id = db.Column(db.Integer, db.ForeignKey('node.id'), nullable=False)
    from_node = db.relationship('Node', foreign_keys=[from_node_id],
                                backref=db.backref('edge_information_froms'))

    to_node_id = db.Column(db.Integer, db.ForeignKey('node.id'), nullable=False)
    to_node = db.relationship('Node', foreign_keys=[to_node_id],
                              backref=db.backref('edge_information_tos'))

    annotation = db.Column(db.Enum(EdgeAnnotation), nullable=False)


class EdgeInformationSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = EdgeInformation
