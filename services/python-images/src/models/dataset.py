from hashlib import blake2b

from marshmallow.validate import Length, OneOf
from marshmallow_sqlalchemy import field_for
from sqlalchemy.sql import func

from src.db import db
from src.master.config import DATA_SOURCE_CONNECTIONS
from src.master.db import data_source_connections
from src.models.base import BaseModel, BaseSchema


def create_dataset_hash(context):
    if context.get_current_parameters()['data_source'] != "postgres":
        session = data_source_connections.get(context.get_current_parameters()['data_source'],
                                              None)
        if session is None:
            return None
    else:
        session = db.session

    load_query = context.get_current_parameters()['load_query']
    result = session.execute(load_query).fetchone()
    num_of_obs = session.execute(f"SELECT COUNT(*) FROM ({load_query}) _subquery_").fetchone()[0]

    hash = blake2b()
    concatenated_result = str(result) + str(num_of_obs)
    hash.update(concatenated_result.encode())

    return str(hash.hexdigest())


class Dataset(BaseModel):
    name = db.Column(db.String)
    description = db.Column(db.String)
    load_query = db.Column(db.String)
    data_source = db.Column(db.String, nullable=False, default="postgres")
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    content_hash = db.Column(db.String, nullable=False, default=create_dataset_hash)


class DatasetSchema(BaseSchema):
    name = field_for(Dataset, 'name', required=True, validate=Length(min=1))
    description = field_for(Dataset, 'description', required=False, allow_none=True, default='')
    load_query = field_for(Dataset, 'load_query', required=True, validate=Length(min=1))
    data_source = field_for(Dataset, 'data_source', validate=OneOf(list(DATA_SOURCE_CONNECTIONS.keys()) + ["postgres"]))

    class Meta(BaseSchema.Meta):
        dump_only = ['time_created', 'content_hash']
        model = Dataset