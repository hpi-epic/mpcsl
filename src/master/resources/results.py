from datetime import datetime

from flask import current_app
from flask_restful import Resource
from marshmallow import Schema, fields

from src.db import db
from src.master.helpers.io import marshal, load_data
from src.models import Job, Result, ResultSchema, Node, Edge


class ResultListResource(Resource):

    def get(self):
        results = Result.query.all()

        return marshal(ResultSchema, results, many=True)
