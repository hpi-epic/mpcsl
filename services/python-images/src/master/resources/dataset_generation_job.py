from src.models import DatasetGenerationJob, DatasetGenerationJobSchema
from src.master.helpers.io import load_data, marshal
from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest
from src.master.helpers.swagger import get_default_response
from flask_restful_swagger_2 import swagger
from flask_restful import Resource, abort, reqparse


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
        dataset_generation_job = DatasetGenerationJob.get_or_404(job_id)
        return marshal(DatasetGenerationJob, dataset_generation_job)
  
    @swagger.doc({
        'description': 'Creates a dataset generation job',
        'parameters': [
          {
            'name': 'dataset generation job input',
            'description': 'Parameters used for dataset generation',
            'in': 'body',
            'schema': DatasetGenerationJobSchema.get_swagger(True)
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
        input_data = load_data(DatasetGenerationJob)

        try:
            dataset_generation_job = DatasetGenerationJob(**input_data)

            db.session.add(dataset_generation_job)
            db.session.commit()
        except DatabaseError:
            raise BadRequest("Could not add dataset generation job to database.")

        return marshal(DatasetGenerationJob, dataset_generation_job)


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

        jobs = DatasetGenerationJob.query.all() if show_hidden else DatasetGenerationJob.query.filter(DatasetGenerationJob.status != DatasetGenerationJob.hidden)

        return marshal(DatasetGenerationJobSchema, jobs, many=True)

