import csv
import io

from flask import Response
from flask_restful import Resource, abort
from flask_restful_swagger_2 import swagger
from marshmallow import Schema, fields
from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest

from src.db import db
from src.master.config import DATA_SOURCE_CONNECTIONS
from src.master.db import data_source_connections
from src.master.helpers.database import add_dataset_nodes
from src.master.helpers.io import load_data, marshal
from src.master.helpers.swagger import get_default_response
from src.models import Dataset, DatasetSchema
from src.models.swagger import SwaggerMixin


class DatasetResource(Resource):
    @swagger.doc({
        'description': 'Returns a single dataset',
        'parameters': [
            {
                'name': 'dataset_id',
                'description': 'Dataset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(DatasetSchema.get_swagger()),
        'tags': ['Dataset']
    })
    def get(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)

        return marshal(DatasetSchema, ds)

    @swagger.doc({
        'description': 'Deletes a dataset',
        'parameters': [
            {
                'name': 'dataset_id',
                'description': 'Dataset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(DatasetSchema.get_swagger()),
        'tags': ['Dataset']
    })
    def delete(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)
        data = marshal(DatasetSchema, ds)

        db.session.delete(ds)
        db.session.commit()
        return data


class DatasetListResource(Resource):
    @swagger.doc({
        'description': 'Returns all available datasets',
        'responses': get_default_response(DatasetSchema.get_swagger().array()),
        'tags': ['Dataset']
    })
    def get(self):
        ds = Dataset.query.all()

        return marshal(DatasetSchema, ds, many=True)

    @swagger.doc({
        'description': 'Creates a dataset',
        'parameters': [
            {
                'name': 'dataset',
                'description': 'Dataset parameters',
                'in': 'body',
                'schema': DatasetSchema.get_swagger(True)
            }
        ],
        'responses': {
            '200': {
                'description': 'Success',
            },
            '400': {
                'description': 'Invalid input data'
            },
            '500': {
                'description': 'Internal server error'
            }
        },
        'tags': ['Dataset']
    })
    def post(self):
        data = load_data(DatasetSchema)

        try:
            ds = Dataset(**data)

            db.session.add(ds)

            add_dataset_nodes(ds)

            db.session.commit()
        except DatabaseError:
            raise BadRequest(f'Could not execute query "{ds.load_query}" on database "{ds.data_source}"')

        return marshal(DatasetSchema, ds)


class DatasetLoadResource(Resource):
    @swagger.doc({
        'description': 'Returns a CSV formatted dataframe that contains the result of the query execution.',
        'parameters': [
            {
                'name': 'dataset_id',
                'description': 'Dataset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': {
            '200': {
                'description': 'Success',
            },
            '404': {
                'description': 'Dataset not found'
            },
            '500': {
                'description': 'Internal server error (likely due to broken query)'
            }
        },
        'produces': ['application/csv'],
        'tags': ['Executor']
    })
    def get(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)

        if ds.data_source != 'postgres':
            session = data_source_connections.get(ds.data_source, None)
            if session is None:
                abort(400)
        else:
            session = db.session

        result = session.execute(ds.load_query)
        keys = [next(filter(lambda n:n.name == name, ds.nodes)).id for name in result.keys()]  # Enforce column order
        result = result.fetchall()

        f = io.StringIO()
        wr = csv.writer(f)
        wr.writerow(keys)
        for line in result:
            wr.writerow(line)
        resp = Response(f.getvalue(), mimetype='text/csv')
        resp.headers.add("X-Content-Length", f.tell())
        return resp


class DataSourceListSchema(Schema, SwaggerMixin):
    data_sources = fields.List(fields.String())


class DatasetAvailableSourcesResource(Resource):
    @swagger.doc({
        'description': 'Returns a list of available data sources.',
        'responses': get_default_response(DataSourceListSchema.get_swagger()),
        'produces': ['application/csv'],
        'tags': ['Executor']
    })
    def get(self):
        val = {
            'data_sources': list(DATA_SOURCE_CONNECTIONS.keys()) + ["postgres"]
        }
        return marshal(DataSourceListSchema, val)
