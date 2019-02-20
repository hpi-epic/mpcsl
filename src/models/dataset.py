from hashlib import blake2b
from marshmallow.validate import Length, OneOf
from marshmallow_sqlalchemy import field_for
from sqlalchemy.sql import func

from src.db import db
from src.master.config import DATA_SOURCE_CONNECTIONS
from src.master.db import data_source_connections
from src.models.base import BaseModel, BaseSchema


def create_dataset_hash(context):
    if context.get_current_parameters()['remote_db'] is not None:
        session = data_source_connections.get(context.get_current_parameters()['remote_db'],
                                              None)
        if session is None:
            return None
    else:
        session = db.session

    result = session.execute(context.get_current_parameters()['load_query'])

    result = result.fetchall()

    hash = blake2b()
    hash.update(str(result).encode())

    return str(hash.hexdigest())


class Dataset(BaseModel):
    name = db.Column(db.String)
    description = db.Column(db.String)
    load_query = db.Column(db.String)
    remote_db = db.Column(db.String)
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    content_hash = db.Column(db.String, nullable=False, default=create_dataset_hash)


class DatasetSchema(BaseSchema):
    name = field_for(Dataset, 'name', required=True, validate=Length(min=1))
    description = field_for(Dataset, 'description', required=False, allow_none=True, default='')
    load_query = field_for(Dataset, 'load_query', required=True, validate=Length(min=1))
    remote_db = field_for(Dataset, 'remote_db', validate=OneOf(list(DATA_SOURCE_CONNECTIONS.keys())))

    class Meta(BaseSchema.Meta):
        dump_only = ['time_created', 'content_hash']
        model = Dataset
