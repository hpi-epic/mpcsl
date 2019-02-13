from src.db import db
from src.models.base import BaseModel, BaseSchema


class Node(BaseModel):
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    dataset = db.relationship('Dataset', backref=db.backref('nodes', cascade="all, delete-orphan"))
    name = db.Column(db.String)


class NodeSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Node
