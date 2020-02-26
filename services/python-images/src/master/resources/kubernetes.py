from flask_restful import Resource
from flask_restful_swagger_2 import swagger
import requests
from src.master.config import SCHEDULER_HOST


class K8SNodeListResource(Resource):
    @swagger.doc({
        'description': 'Returns all cluster nodes',
        'responses': {
            '200': {
                'description': 'K8sNodes',
                'examples': {
                    'application/json': ['minikube']
                }
            }
        }
    })
    def get(self):
        resp = requests.get(f'http://{SCHEDULER_HOST}/api/nodes')
        return resp.json()
