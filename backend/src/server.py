from io import StringIO

import numpy as np
import pandas as pd
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/datasets/<dataset_id>', methods=['GET'])
def dataset(dataset_id):
    mean = [0, 5, 10]
    cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
    ds = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=1000), columns=['X1', 'X2', 'X3'])
    return ds.to_csv(index=False)


@app.route('/results', methods=['POST'])
def results():
    df = pd.read_csv(StringIO(str(request.data)))
    app.logger.info(df)
    return 'OK'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
