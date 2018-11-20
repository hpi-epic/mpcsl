from io import StringIO

import pandas as pd
from flask import current_app, request
from flask_restful import Resource


class Results(Resource):

    def post(self):
        df = pd.read_csv(StringIO(str(request.data, encoding='utf-8')))
        current_app.logger.info(df)
        return 'OK'
