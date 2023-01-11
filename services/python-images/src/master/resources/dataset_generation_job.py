from sqlalchemy import func

from src.master.helpers.socketio_events import job_status_change
from src.models import DatasetGenerationJob, DatasetGenerationJobSchema, Dataset, DatasetSchema
from src.master.helpers.io import load_data, marshal
from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest
from src.master.helpers.swagger import get_default_response
from flask_restful_swagger_2 import swagger
from flask_restful import Resource, reqparse, abort
from src.db import db
from src.models.job import JobStatus
from flask import request
import pandas as pd
import uuid
from pandas.errors import ParserError
from src.master.config import DB_DATABASE
from src.master.helpers.database import add_dataset_nodes
from marshmallow import fields
from src.models import BaseSchema


class DatasetGenerationJobResource(Resource):
    @swagger.doc({
        'description': 'Get a data generation job with all parameters used for data set generation.',
        'parameters': [
          {
            'name': 'job_id',
            'description': 'Dataset job generation identifier.',
            'in': 'path',
            'type': 'integer',
            'required': True
          }
        ],
        'responses': get_default_response(DatasetGenerationJobSchema.get_swagger()),
        'tags': ['DatasetGenerationJob', 'Job']
    })
    def get(self, job_id):
        dataset_generation_job = DatasetGenerationJob.query.get_or_404(job_id)
        return marshal(DatasetGenerationJobSchema, dataset_generation_job)

    @swagger.doc({
      'description': 'Creates a new table from the uploaded csv file',
      'parameters': [
        {
          'name': 'job_id',
          'description': 'Dataset job generation identifier.',
          'in': 'path',
          'type': 'integer',
          'required': True
        },
        {
            "name": "file",
            "in": "formData",
            "description": "file in csv format to upload",
            "required": True,
            "type": "file"
        }
      ],
      'responses': {
            '200': {
                'description': 'Success',
            },
            '400': {
                'description': 'Attached file was not in correct csv format'
            },
            '500': {
                'description': 'Internal server error (likely due to broken query)'
            }
      },
      'tags': ['DatasetGenerationJob', 'Job']
    })
    def put(self, job_id):
        if 'file' not in request.files:
            abort(400, message='no file attached')

        if not job_id:
            abort(400, message='missing job_id')

        job: DatasetGenerationJob = DatasetGenerationJob.query.get_or_404(job_id)

        if job.dataset:
            abort(400, message='Dataset generation job already mapped')

        file = request.files['file']
        # The char - is not allowed in sqlAlchemy
        sql_conform_id = str(uuid.uuid4()).replace('-', '_')
        table_name = job.datasetName + '_' + sql_conform_id
        try:
            data = pd.read_csv(file)
            data.to_sql(table_name, db.engine, index=False)
        except ParserError as e:
            abort(400, message=f'Invalid format: {e}')

        dataset = Dataset(
            description="generated",
            load_query=f"SELECT * FROM {table_name}",
            data_source=DB_DATABASE,
            name=job.datasetName
        )
        db.session.add(dataset)

        job.dataset = dataset
        job.status = JobStatus.done

        job.end_time = func.now()

        add_dataset_nodes(dataset)

        db.session.commit()
        job_status_change(job, None)
        return marshal(DatasetSchema, dataset)


class DatasetGenerationJobInputSchema(BaseSchema):
    nodes = fields.Integer()
    samples = fields.Integer()
    edgeProbability = fields.Float()  # TODO add range 0 - 1
    edgeValueLowerBound = fields.Float()
    edgeValueUpperBound = fields.Float()
    datasetName = fields.String()
    kubernetesNode = fields.String()


class DatasetGenerationJobListResource(Resource):
    @swagger.doc({
        'description': 'Returns all dataset generation jobs',
        'parameters': [
            {
                'name': 'show_hidden',
                'description': 'Pass show_hidden=1 to display also hidden jobs',
                'in': 'query',
                'type': 'integer',
                'enum': [0, 1],
                'default': 0
            }
        ],
        'responses': get_default_response(DatasetGenerationJobSchema.get_swagger().array()),
        'tags': ['DatasetGenerationJob', 'Job']
    })
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('show_hidden', required=False, type=int, store_missing=False)
        show_hidden = parser.parse_args().get('show_hidden', 0) == 1

        jobs = DatasetGenerationJob.query.all() \
            if show_hidden \
            else DatasetGenerationJob.query.filter(DatasetGenerationJob.status != JobStatus.hidden)

        return marshal(DatasetGenerationJobSchema, jobs, many=True)

    @swagger.doc({
        'description': 'Creates a dataset generation job',
        'parameters': [
            {
                'name': 'dataset generation job input',
                'description': 'Parameters used for dataset generation',
                'in': 'body',
                'schema': DatasetGenerationJobInputSchema.get_swagger()
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
        'tags': ['DatasetGenerationJob', 'Job']
    })
    def post(self):
        input_data = load_data(DatasetGenerationJobSchema)

        try:
            dataset_generation_job = DatasetGenerationJob(**input_data)
            dataset_generation_job.node_hostname = request.json.get("kubernetesNode")

            db.session.add(dataset_generation_job)
            db.session.commit()
        except DatabaseError:
            raise BadRequest("Could not add dataset generation job to database.")

        return marshal(DatasetGenerationJobSchema, dataset_generation_job)
