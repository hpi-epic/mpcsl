from flask import request
from flask_restful import Resource, abort
from flask_restful_swagger_2 import swagger
import pandas as pd
import uuid
from src.db import db
from pandas.io.parsers import ParserError

from src.master.config import DB_DATABASE
from src.master.helpers.database import add_dataset_nodes
from src.models import Dataset


class DatasetCsvUploadResource(Resource):
    @swagger.doc({
        'description': 'Creates a new table from the uploaded csv file',
        'parameters': [
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
        'tags': ['Dataset']
    })
    def post(self):
        if 'file' not in request.files:
            abort(400, message='no file attached')
        file = request.files['file']
        # The char - is not allowed in sqlAlchemy
        sql_conform_id = str(uuid.uuid4()).replace('-', '_')
        table_name = "generated" + sql_conform_id
        try:
            data = pd.read_csv(file, index_col=0)
            data.to_sql(table_name, db.engine, index=False)
        except ParserError as e:
            abort(400, message=f'Invalid format: {e}')

        dataset = Dataset(
            description="generated",
            load_query=f"SELECT * FROM {table_name}",
            data_source=DB_DATABASE,
            name=table_name
        )
        db.session.add(dataset)

        add_dataset_nodes(dataset)
