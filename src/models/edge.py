from src.db import db
from src.models.base import BaseModel, BaseSchema


class Edge(BaseModel):

    result_id = db.Column(db.Integer, db.ForeignKey('result.id'))
    result = db.relationship('Result')

    from_node_id = db.Column(db.Integer, db.ForeignKey('node.id'))
    from_node = db.relationship('Node', foreign_keys=[from_node_id])

    to_node_id = db.Column(db.Integer, db.ForeignKey('node.id'))
    to_node = db.relationship('Node', foreign_keys=[to_node_id])


class EdgeSchema(BaseSchema):
    class Meta:
        model = Edge
