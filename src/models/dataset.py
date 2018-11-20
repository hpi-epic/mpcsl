from src.db import db
from src.models.base import BaseModel, BaseSchema


class Dataset(BaseModel):
    query = db.Column(db.String, required=True)
    name = db.Column(db.String, required=True)


class DatasetSchema(BaseSchema):
    class Meta:
        model = Dataset
