from flask_restful import Resource
from src.master.helpers.io import marshal
from src.models import Result, ResultSchema


class ResultListResource(Resource):

    def get(self):
        results = Result.query.all()

        return marshal(ResultSchema, results, many=True)
