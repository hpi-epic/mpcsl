import os
import requests
import pandas as pd
import numpy as np
import signal
import time
from unittest import TestCase
from multiprocessing import Process
from urllib.error import URLError
from urllib.request import urlopen

from src.master.appfactory import AppFactory
from src.db import db
from src.master.executor import ExecutorResource
from src.models import Result, Job
from test.factories import DatasetFactory


class BaseIntegrationTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.factory = AppFactory()
        cls.app = cls.factory.up()
        cls.api = cls.factory.api
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        cls.db = db
        cls.original_tables = cls.db.metadata.sorted_tables

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()

    def setUp(self):
        def run_func(app):
            app.run(host="0.0.0.0", port='5000', debug=True, use_reloader=False, threaded=True)
        self.app_thread = Process(target=run_func, args=(self.app, ))

        self.db.create_all()

        self.app_thread.start()
        timeout = 5
        while timeout > 0:
            time.sleep(1)
            try:
                urlopen('localhost:5000')
                timeout = 0
            except URLError:
                timeout -= 1

    def tearDown(self):
        self.stop_app_thread()
        self.db.session.remove()
        self.db.reflect()
        self.drop_all()

    def drop_all(self):
        for tbl in reversed(self.db.metadata.sorted_tables):
            tbl.drop(self.db.engine)
            if tbl not in self.original_tables:
                self.db.metadata.remove(tbl)

    def stop_app_thread(self):
        if self.app_thread:
            if self._stop_cleanly():
                return
            if self.app_thread.is_alive():
                self.app_thread.terminate()

    def _stop_cleanly(self, timeout=5):
        try:
            os.kill(self.app_thread.pid, signal.SIGINT)
            self.app_thread.join(timeout)
            return True
        except Exception as ex:
            print('Failed to join the live server process: {}'.format(ex))
            return False

    def url_for(self, resource, **values):
        adapter = self.app.url_map.bind('localhost:5000')
        return adapter.build(resource.endpoint, values, force_external=True)

    @staticmethod
    def setup_dataset_gauss():
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
        return ds

    @staticmethod
    def setup_dataset_discrete():
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                a int,
                b int,
                c int
            );
        """)
        source = np.random.randint(0, 10, (50, 3))
        for l in source:
            db.session.execute("INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in l])))
        return ds

    @staticmethod
    def setup_dataset_binary():
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                a int,
                b int,
                c int
            );
        """)
        source = np.random.randint(0, 2, (50, 3))
        for l in source:
            db.session.execute("INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in l])))
        return ds

    @staticmethod
    def setup_dataset_cooling_house():
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        df = pd.read_csv('test/fixtures/coolinghouse_1k.csv', index_col=0)
        df.to_sql('test_data', con=db.engine, index=False)
        return ds

    def run_experiment(self, experiment):
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=experiment.id))
        assert job_r.status_code == 200

        job = db.session.query(Job).get(job_r.json()['id'])

        result = None
        i = 0
        while result is None:
            if i > 15:
                raise TimeoutError
            time.sleep(1)

            # If this fails because of transaction abort, check R script (use same session)
            result = db.session.query(Result).filter(Result.job == job).first()
            i += 1
        return job, result
