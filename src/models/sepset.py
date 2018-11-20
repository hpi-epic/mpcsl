from src.db import db
from src.models.base import BaseModel, BaseSchema


class SepSet(BaseModel):
    result_id = db.Column(db.Integer, db.ForeignKey('result.id'), nullable=False)
    result = db.relationship('Result')

    nodes = db.Column(db.ARRAY(db.Integer), nullable=False)
    statistic = db.Column(db.Float, nullable=False)
    level = db.Column(db.Integer, nullable=False)


class SepSetSchema(BaseSchema):
    class Meta:
        model = SepSet
