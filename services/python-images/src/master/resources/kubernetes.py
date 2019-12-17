from flask_restful import Resource
import requests
from src.master.config import SCHEDULER_HOST


class K8SNodeListResource(Resource):
    def get(self):
        resp = requests.get(f'http://{SCHEDULER_HOST}/api/nodes')
        return resp.json()
