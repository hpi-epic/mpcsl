from io import StringIO

import pandas as pd
from flask import current_app, request
from flask_restful import Resource


class Results(Resource):

    def post(self):
        result = '\n'.join(request.json['result'])
        current_app.logger.info(result)
        df = pd.read_csv(StringIO(result))
        # current_app.logger.info(df)
        return 'OK'
