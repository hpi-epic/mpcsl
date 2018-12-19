from marshmallow.validate import Length
from marshmallow_sqlalchemy import field_for
from sqlalchemy.sql import func

from src.db import db
from src.models.base import BaseModel, BaseSchema


class Dataset(BaseModel):
    name = db.Column(db.String)
    description = db.Column(db.String)
    load_query = db.Column(db.String)
    remote_db = db.Column(db.String)
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)


class DatasetSchema(BaseSchema):
    name = field_for(Dataset, 'name', required=True, validate=Length(min=1))
    description = field_for(Dataset, 'name', required=False, allow_none=True, default='')
    load_query = field_for(Dataset, 'name', required=True, validate=Length(min=1))
    remote_db = field_for(Dataset, 'name', required=True, allow_none=True, validate=Length(min=1))

    class Meta(BaseSchema.Meta):
        dump_only = ['time_created']
        model = Dataset
