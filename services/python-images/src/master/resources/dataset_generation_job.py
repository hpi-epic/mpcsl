from src.models import DatasetGenerationJob, DatasetGenerationJobSchema
from src.master.helpers.io import load_data, marshal
from marshmallow import Schema, fields
from src.models.swagger import SwaggerMixin
from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest

class DatasetGenerationJobResource:
  @swagger.doc({
    'description': 'Get a data generation job with all parameters used for data set generation.'
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
    'description': 'Deletes a dataset generation job. Stops and deletes a running job.',
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
  def delete(self, job_id):
    dataset_generation_job = DatasetGenerationJob.query.get_or_404(job_id)
    data = marshal(DatasetGenerationJob, dataset_generation_job)

    db.session.delete(dataset_generation_job)
    db.session.commit()
    return data
  
  @swagger.doc({
    'description': 'Creates a dataset generation job',
    'parameters': [
        {
            'name': 'dataset generation job input',
            'description': 'Parameters used for dataset generation',
            'in': 'body',
            'schema': DatasetGenerationJobSchema
            .get_swagger(True)
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
