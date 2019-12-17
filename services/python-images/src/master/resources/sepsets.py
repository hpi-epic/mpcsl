from flask_restful import Resource
from flask_restful_swagger_2 import swagger

from src.master.helpers.io import marshal
from src.master.helpers.swagger import get_default_response
from src.models import Sepset, SepsetSchema


class SepsetResource(Resource):
    @swagger.doc({
        'description': 'Returns a single sepset',
        'parameters': [
            {
                'name': 'sepset_id',
                'description': 'Sepset identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(SepsetSchema.get_swagger()),
        'tags': ['Sepset']
    })
    def get(self, sepset_id):
        sepset = Sepset.query.get_or_404(sepset_id)

        return marshal(SepsetSchema, sepset)


class ResultSepsetListResource(Resource):
    @swagger.doc({
        'description': 'Returns all sepsets for one result',
        'parameters': [
            {
                'name': 'result_id',
                'description': 'Result identifier',
                'in': 'path',
                'type': 'integer',
                'required': True
            }
        ],
        'responses': get_default_response(SepsetSchema.get_swagger().array()),
        'tags': ['Sepset']
    })
    def get(self, result_id):
        sepsets = Sepset.query.filter(Sepset.result_id == result_id).all()

        return marshal(SepsetSchema, sepsets, many=True)
