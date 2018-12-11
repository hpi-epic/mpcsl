from sqlalchemy.sql import func

from src.db import db
from src.models.base import BaseModel, BaseSchema


class Dataset(BaseModel):
    load_query = db.Column(db.String)
    name = db.Column(db.String)
    description = db.Column(db.String)
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)


class DatasetSchema(BaseSchema):
    class Meta(BaseSchema.Meta):
        model = Dataset
