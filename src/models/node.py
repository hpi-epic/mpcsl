from src.db import db
from src.models.base import BaseModel, BaseSchema


class Node(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'), nullable=False)
    result = db.relationship('Result', backref=db.backref('nodes', cascade="all, delete-orphan"))
    name = db.Column(db.String)


class NodeSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Node
