import csv
import io
from io import StringIO

import networkx as nx
import logging

from flask_restful_swagger_2 import swagger
from flask import Response, request
from flask_restful import Resource, abort
from marshmallow import Schema, fields
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest

from src.db import db
from src.master.config import DATA_SOURCE_CONNECTIONS
from src.master.db import data_source_connections
from src.master.helpers.database import add_dataset_nodes
from src.master.helpers.io import load_data, marshal
from src.master.helpers.swagger import get_default_response
from src.models import Dataset, DatasetSchema, Edge, ExperimentSchema

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

    @swagger.doc({
        'description': 'Updates a dataset',
        'parameters': [
            {
                'name': 'dataset_id',
                'description': 'Dataset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            },
            {
                'name': 'dataset',
                'description': 'Dataset parameters. Only description is editable',
                'in': 'body',
                'schema': DatasetSchema.get_swagger(True)
            }
        ],
        'responses': get_default_response(DatasetSchema.get_swagger()),
        'tags': ['Dataset']
    })
    def put(self, dataset_id):
        description = request.json.get('description')
        dataset = Dataset.query.get_or_404(dataset_id)
        if description:
            dataset.description = description
        else:
            raise BadRequest('Body must contain description')

        db.session.commit()

        return marshal(DatasetSchema, dataset)


class DatasetMetadataSchema(Schema, SwaggerMixin):
    variables = fields.Integer()
    time_created = fields.DateTime()
    observations = fields.Integer()
    data_source = fields.String()
    query = fields.String()
    has_ground_truth = fields.Boolean()


# Memory caching of metadata responses. Should be optimized in case of many datasets
metadata_cache = {}


class DatasetMetadataResource(Resource):

    @swagger.doc({
        'description': 'Returns metadata of a single dataset',
        'parameters': [
            {
                'name': 'dataset_id',
                'description': 'Dataset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(DatasetMetadataSchema.get_swagger()),
        'tags': ['Dataset']
    })
    def get(self, dataset_id):
        try:
            data = metadata_cache[str(dataset_id)]
            logging.info(f'Using cached metadata for {dataset_id}')
            return data
        except KeyError:
            ds: Dataset = Dataset.query.get_or_404(dataset_id)
            data = ds.ds_metadata()
            metadata_cache[str(dataset_id)] = data
            return data


def parse_graph(file: FileStorage, is_from_igraph_r_package: bool = False) -> nx.Graph:
    # The Igraph package in R stores in a weird version where the label is stored under a tag "name"
    try:
        if is_from_igraph_r_package:
            graph = nx.parse_gml(file.stream.read().decode('utf-8'), label="name")
        else:
            graph = nx.parse_gml(file.stream.read().decode('utf-8'))
    finally:
        file.stream.seek(0)
    return graph


class DatasetGroundTruthUploadResource(Resource):
    @swagger.doc({
        'description': 'Add Ground-Truth to Dataset',
        'parameters': [
            {
                'name': 'dataset_id',
                'description': 'Dataset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }, {
                'name': 'graph_file',
                'description': 'Path to Graph File',
                'in': 'formData',
                'type': 'file',
                'required': True
            }
        ],
        'responses': get_default_response(DatasetSchema.get_swagger()),
        'tags': ['Dataset']
    })
    def post(self, dataset_id):
        try:
            file = request.files['graph_file']
        except Exception:
            raise BadRequest("No file graph_file attached")
        try:
            graph = parse_graph(file)
        except Exception:
            try:
                graph = parse_graph(file, is_from_igraph_r_package=True)
            except Exception:
                raise BadRequest(f'Could not parse file: "{file.filename}"')
        ds = Dataset.query.get_or_404(dataset_id)
        for node in ds.nodes:
            for edge in node.edge_froms:
                if edge.is_ground_truth:
                    db.session.delete(edge)
        db.session.flush()
        for edge in graph.edges:
            from_node_label = edge[0]
            to_node_label = edge[1]
            from_node_index = None
            to_node_index = None

            for node in ds.nodes:
                if from_node_label == node.name:
                    from_node_index = node.id
                if to_node_label == node.name:
                    to_node_index = node.id
            if from_node_index is None:
                raise BadRequest(f'No node with Label: "{from_node_label}" in {ds.name}')
            if to_node_index is None:
                raise BadRequest(f'No node with Label: "{to_node_label}" in {ds.name}')
            edge = Edge(result_id=None, from_node_id=from_node_index,
                        to_node_id=to_node_index, weight=None, is_ground_truth=True)
            db.session.add(edge)

        db.session.commit()
        try:
            del metadata_cache[str(dataset_id)]
        except KeyError:
            pass
        # What to return?
        return marshal(DatasetSchema, ds)


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
            db.session.commit()

            add_dataset_nodes(ds)
        except DatabaseError:
            raise BadRequest(f'Could not execute query "{ds.load_query}" on database "{ds.data_source}"')

        return marshal(DatasetSchema, ds)


def load_dataset_as_csv(session: Session, ds: Dataset, with_ids: bool = True) -> StringIO:
    result = session.execute(ds.load_query)
    if with_ids:
        keys = [next(filter(lambda n: n.name == name, ds.nodes)).id for name in result.keys()]  # Enforce column order
    else:
        keys = [next(filter(lambda n: n.name == name, ds.nodes)).name for name in result.keys()]  # Enforce column order
    result = result.fetchall()

    f = io.StringIO()
    wr = csv.writer(f)
    wr.writerow(keys)
    for line in result:
        wr.writerow(line)
    f.seek(0)
    return f


class DatasetLoadResource(Resource):
    @swagger.doc({
        'description': 'Returns a CSV formatted dataframe that contains the result of the query execution \
                        with names of the columns.',
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
    def get(self, dataset_id, by_id=0):
        ds = Dataset.query.get_or_404(dataset_id)

        if ds.data_source != 'postgres':
            session = data_source_connections.get(ds.data_source, None)
            if session is None:
                abort(400)
        else:
            session = db.session
        f = load_dataset_as_csv(session, ds, with_ids=False)

        resp = Response(f.getvalue(), mimetype='text/csv')
        resp.headers.add("X-Content-Length", f.tell())
        return resp


class DatasetLoadWithIdsResource(Resource):
    @swagger.doc({
        'description': 'Returns a CSV formatted dataframe that contains the result of the query execution \
                        with ids of the columns.',
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
    def get(self, dataset_id, by_id=0):
        ds = Dataset.query.get_or_404(dataset_id)

        if ds.data_source != 'postgres':
            session = data_source_connections.get(ds.data_source, None)
            if session is None:
                abort(400)
        else:
            session = db.session

        f = load_dataset_as_csv(session, ds, with_ids=True)

        resp = Response(f.getvalue(), mimetype='text/csv')
        resp.headers.add("X-Content-Length", f.tell())
        return resp


class DatasetExperimentResource(Resource):
    @swagger.doc({
        'description': 'Returns all experiments belonging to this dataset',
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
        'tags': ['Executor']
    })
    def get(self, dataset_id):
        ds = Dataset.query.get_or_404(dataset_id)
        print(ds)
        return marshal(ExperimentSchema, ds.experiments, many=True)


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
