from src.db import db
from src.models.base import BaseModel, BaseSchema


class Sepset(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'), nullable=False)
    result = db.relationship('Result')

    from_node_id = db.Column(db.Integer, db.ForeignKey('node.id'))
    from_node = db.relationship('Node', foreign_keys=[from_node_id])

    to_node_id = db.Column(db.Integer, db.ForeignKey('node.id'))
    to_node = db.relationship('Node', foreign_keys=[to_node_id])

    nodes = db.Column(db.ARRAY(db.Integer), nullable=False)
    statistic = db.Column(db.Float, nullable=False)
    level = db.Column(db.Integer, nullable=False)


class SepsetSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Sepset
