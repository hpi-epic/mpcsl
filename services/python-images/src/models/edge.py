from src.db import db
from src.models.base import BaseModel, BaseSchema


class Edge(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'), nullable=True)
    result = db.relationship('Result', backref=db.backref('edges', cascade="all, delete-orphan"))

    from_node_id = db.Column(db.Integer, db.ForeignKey('node.id'), nullable=False)
    from_node = db.relationship('Node', foreign_keys=[from_node_id],
                                backref=db.backref('edge_froms'))

    to_node_id = db.Column(db.Integer, db.ForeignKey('node.id'), nullable=False)
    to_node = db.relationship('Node', foreign_keys=[to_node_id],
                              backref=db.backref('edge_tos'))

    weight = db.Column(db.Float)

    is_ground_truth = db.Column(db.Boolean)


class EdgeSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Edge
