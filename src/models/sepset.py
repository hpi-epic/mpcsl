from src.db import db
from src.models.base import BaseModel, BaseSchema


class Sepset(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'), nullable=False)
    result = db.relationship('Result', backref=db.backref('sepsets', cascade="all, delete-orphan"))

    from_node_id = db.Column(db.Integer, db.ForeignKey('node.id'), nullable=False)
    from_node = db.relationship('Node', foreign_keys=[from_node_id],
                                backref=db.backref('sepset_froms', cascade="all, delete-orphan"))

    to_node_id = db.Column(db.Integer, db.ForeignKey('node.id'), nullable=False)
    to_node = db.relationship('Node', foreign_keys=[to_node_id],
                              backref=db.backref('sepset_tos', cascade="all, delete-orphan"))

#    node_names = db.Column(db.ARRAY(db.String), nullable=False)
    statistic = db.Column(db.Float, nullable=False)
    level = db.Column(db.Integer, nullable=False)


class SepsetSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Sepset
