from src.db import db
from src.models.base import BaseModel, BaseSchema


class DataSet(BaseModel):
    query = db.Column(db.String, required=True)
    name = db.Column(db.String, required=True)


class DataSetSchema(BaseSchema):
    class Meta:
        model = DataSet
