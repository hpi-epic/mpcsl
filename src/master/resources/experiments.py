from flask_restful import Resource


class Experiments(Resource):
    def get(self):
        return 'Ok'
