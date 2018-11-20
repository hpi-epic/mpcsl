from src.db import db
from src.models.base import BaseModel, BaseSchema


class Node(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'))
    result = db.relationship('Result')
    name = db.Column(db.String)


class NodeSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Node