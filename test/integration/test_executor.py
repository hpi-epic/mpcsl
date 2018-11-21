import time

import numpy as np
import requests

from src.db import db
from src.models import Result, Job, Node
from src.master.executor import Executor
from test.factories import ExperimentFactory, DatasetFactory
from .base import BaseIntegrationTest


class ExecutorTest(BaseIntegrationTest):

    def test_r_execution(self):
        # Set up fixtures
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                a float,
                b float,
                c float
            );
        """)
        mean = [0, 5, 10]
        cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
        source = np.random.multivariate_normal(mean, cov, size=50)
        for l in source:
            db.session.execute("INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in l])))
        ex = ExperimentFactory(dataset=ds)
        db.session.commit()

        # When
        job = requests.get(self.api.url_for(Executor, experiment_id=ex.id)).json()
        job = db.session.query(Job).get(job['id'])

        result = None
        i = 0
        while result is None:
            if i > 30:
                raise TimeoutError
            time.sleep(1)

            # If this fails because of transaction abort, check R script (use same session)
            result = db.session.query(Result).first()
            i += 1

        assert result.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
