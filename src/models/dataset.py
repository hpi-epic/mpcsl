from src.db import db
from src.models.base import BaseModel, BaseSchema


class Dataset(BaseModel):
    load_query = db.Column(db.String)
    name = db.Column(db.String)


class DatasetSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Dataset
