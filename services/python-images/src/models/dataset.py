from marshmallow.validate import Length, OneOf
from marshmallow_sqlalchemy import field_for
from sqlalchemy.sql import func

from src.db import db
from src.master.config import DATA_SOURCE_CONNECTIONS
from src.master.db import data_source_connections
from src.models.base import BaseModel, BaseSchema
from src.mater.helpers.database import create_data_hash

from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest


def create_dataset_hash(context):
    if context.get_current_parameters()['data_source'] != "postgres":
        session = data_source_connections.get(context.get_current_parameters()['data_source'],
                                              None)
        if session is None:
            return None
    else:
        session = db.session

    load_query = context.get_current_parameters()['load_query']

    return create_data_hash(session, load_query=load_query)


class Dataset(BaseModel):
    name = db.Column(db.String)
    description = db.Column(db.String)
    load_query = db.Column(db.String)
    data_source = db.Column(db.String, nullable=False, default="postgres")
    time_created = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    content_hash = db.Column(db.String, nullable=False, default=create_dataset_hash)

    def ds_metadata(self):
        already_has_ground_truth = False
        for node in self.nodes:
            for edge in node.edge_froms:
                if edge.is_ground_truth:
                    already_has_ground_truth = True
                    break
            if already_has_ground_truth:
                break

        if self.data_source != "postgres":
            session = data_source_connections.get(self.data_source, None)
            if session is None:
                raise BadRequest(f'Could not reach database "{self.data_source}"')
        else:
            session = db.session

        try:
            num_of_obs = session.execute(f"SELECT COUNT(*) FROM ({self.load_query}) _subquery_").fetchone()[0]
        except DatabaseError:
            raise BadRequest(f'Could not execute query "{self.load_query}" on database "{self.data_source}"')
        data = {
            'variables': len(self.nodes),
            'time_created': self.time_created.timestamp(),
            'observations': int(num_of_obs),
            'data_source': self.data_source,
            'query': self.load_query,
            'has_ground_truth': already_has_ground_truth
        }
        return data


class DatasetSchema(BaseSchema):
    name = field_for(Dataset, 'name', required=True, validate=Length(min=1))
    description = field_for(Dataset, 'description', required=False, allow_none=True, default='')
    load_query = field_for(Dataset, 'load_query', required=True, validate=Length(min=1))
    data_source = field_for(Dataset, 'data_source', validate=OneOf(list(DATA_SOURCE_CONNECTIONS.keys()) + ["postgres"]))

    class Meta(BaseSchema.Meta):
        dump_only = ['time_created', 'content_hash']
        model = Dataset
