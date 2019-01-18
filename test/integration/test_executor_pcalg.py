import time

import numpy as np
import pandas as pd
import pytest
import requests

from src.db import db
from src.models import Result, Job, Node, Edge, Sepset
from src.master.executor import ExecutorResource
from test.factories import ExperimentFactory, DatasetFactory
from .base import BaseIntegrationTest


class GaussExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-4)
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
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))
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

        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3


class DiscreteExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-3)
    def test_r_execution_discrete(self):
        # Set up fixtures
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
        ex = ExperimentFactory(dataset=ds)
        ex.parameters['independence_test'] = 'disCI'
        db.session.commit()

        # When
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))
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

        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3


class BinaryExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-2)
    def test_r_execution_binary(self):
        # Set up fixtures
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
        ex = ExperimentFactory(dataset=ds)
        ex.parameters['independence_test'] = 'binCI'
        db.session.commit()

        # When
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))
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

        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3


class SepsetExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-1)
    def test_r_execution_with_sepsets(self):
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        df = pd.read_csv('test/fixtures/coolinghouse_1k.csv', index_col=0)
        df.to_sql('test_data', con=db.engine, index=False)
        ex = ExperimentFactory(dataset=ds)
        ex.parameters['alpha'] = 0.05
        db.session.commit()

        # When
        job_r = requests.post(self.url_for(ExecutorResource, experiment_id=ex.id))
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

        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).all()
        node_set = {'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
        for node in nodes:
            assert node.name in node_set
            node_set.remove(node.name)
        assert len(node_set) == 0

        edges = db.session.query(Edge).all()
        edge_set = {('V1', 'V3'), ('V2', 'V3'), ('V4', 'V2'), ('V4', 'V5'), ('V4', 'V6'),
                    ('V5', 'V4'), ('V6', 'V4')}
        for edge in edges:
            assert (edge.from_node.name, edge.to_node.name) in edge_set
            edge_set.remove((edge.from_node.name, edge.to_node.name))
        assert len(edge_set) == 0

        sepsets = db.session.query(Sepset).all()
        sepset_set = [('V4', 'V1', ['V5']), ('V4', 'V3', ['V5', 'V6']),
                      ('V5', 'V1', ['V4']), ('V5', 'V2', ['V4']), ('V5', 'V3', ['V4']),
                      ('V6', 'V1', ['V4']), ('V6', 'V2', ['V3', 'V4']),
                      ('V6', 'V3', ['V1', 'V4']), ('V6', 'V5', ['V4'])]
        for sepset in sepsets:
            assert (sepset.from_node.name, sepset.to_node.name, sepset.node_names) in sepset_set
        assert len(sepset_set) == len(sepsets)