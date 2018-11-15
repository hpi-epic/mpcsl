from io import StringIO

import pandas as pd
from flask import current_app, request
from flask_restful import Resource


class Results(Resource):

    def __init__(self, **kwargs):
        self.logger = kwargs['logger']

    def post(self):
        df = pd.read_csv(StringIO(str(request.data)))
        current_app.logger.info(df)
        return 'OK'
