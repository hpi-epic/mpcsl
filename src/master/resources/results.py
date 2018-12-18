from flask_restful import Resource
from flask_restful_swagger_2 import swagger

from src.master.helpers.io import marshal
from src.master.helpers.swagger import get_default_response
from src.models import Result, ResultSchema


class ResultListResource(Resource):

    @swagger.doc({
        'description': 'Returns all available results',
        'responses': get_default_response(ResultSchema.get_swagger().array()),
        'tags': ['Result']
    })
    def get(self):
        results = Result.query.all()

        return marshal(ResultSchema, results, many=True)
