import os

import pytest
import requests

from src.db import db
from src.master.resources import JobLogsResource
from src.master.helpers.io import get_logfile_name
from src.models import Node, Sepset, Edge
from test.factories import ExperimentFactory
from .base import BaseIntegrationTest


class GaussExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-4)
    def test_r_execution_gauss(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_gauss())
        db.session.commit()

        # When
        job, result = self.run_experiment(ex)

        # Then
        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).filter_by(dataset_id=job.experiment.dataset_id).all()
        edges = db.session.query(Edge).filter_by(result=result).all()
        for edge in edges:
            assert edge.from_node in nodes
            assert edge.from_node in nodes
        assert len(edges) > 0  # TODO


class DiscreteExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-3)
    def test_r_execution_discrete(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_discrete())
        ex.parameters['independence_test'] = 'disCI'
        db.session.commit()

        # When
        job, result = self.run_experiment(ex)

        # Then
        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).filter_by(dataset_id=job.experiment.dataset_id).all()
        edges = db.session.query(Edge).filter_by(result=result).all()
        for edge in edges:
            assert edge.from_node in nodes
            assert edge.from_node in nodes
        assert len(edges) > 0  # TODO


class BinaryExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-2)
    def test_r_execution_binary(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_binary())
        ex.parameters['independence_test'] = 'binCI'
        db.session.commit()

        # When
        job, result = self.run_experiment(ex)

        # Then
        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).filter_by(dataset_id=job.experiment.dataset_id).all()
        edges = db.session.query(Edge).filter_by(result=result).all()
        for edge in edges:
            assert edge.from_node in nodes
            assert edge.from_node in nodes
        assert len(edges) > 0  # TODO


class SepsetExecutorTest(BaseIntegrationTest):

    PATCHES = {'src.master.resources.jobs.LOAD_SEPARATION_SET': True,
               'src.master.executor.executor.LOAD_SEPARATION_SET': True}

    @pytest.mark.run(order=-1)
    def test_r_execution_with_sepsets(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_cooling_house())
        ex.parameters['alpha'] = 0.05
        db.session.commit()

        # When
        job, result = self.run_experiment(ex)

        # Then
        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        nodes = db.session.query(Node).filter_by(dataset_id=job.experiment.dataset_id).all()
        node_set = {'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
        for node in nodes:
            assert node.name in node_set
            node_set.remove(node.name)
        assert len(node_set) == 0

        edges = db.session.query(Edge).filter_by(result=result).all()
        edge_set = {
            ('V1', 'V3'): 0.4803,
            ('V2', 'V3'): 0.2127,
            ('V4', 'V2'): 0.8004,
            ('V4', 'V5'): 1.5227,
            ('V4', 'V6'): 1.5653,
            ('V5', 'V4'): 1.5227,
            ('V6', 'V4'): 1.5653
        }
        for edge in edges:
            from_name, to_name = edge.from_node.name, edge.to_node.name

            assert (from_name, to_name) in edge_set
            self.assertAlmostEqual(edge.weight, edge_set[(from_name, to_name)], places=2)

            del edge_set[(edge.from_node.name, edge.to_node.name)]
        assert len(edge_set) == 0

        sepsets = db.session.query(Sepset).all()
        sepset_set = [('V4', 'V1', ['V5']), ('V4', 'V3', ['V5', 'V6']),
                      ('V5', 'V1', ['V4']), ('V5', 'V2', ['V4']), ('V5', 'V3', ['V4']),
                      ('V6', 'V1', ['V4']), ('V6', 'V2', ['V3', 'V4']),
                      ('V6', 'V3', ['V1', 'V4']), ('V6', 'V5', ['V4'])]
        for sepset in sepsets:
            assert (sepset.from_node.name, sepset.to_node.name) or \
                   (sepset.to_node.name, sepset.from_node.name) in sepset_set
        assert len(sepset_set) == len(sepsets)


class ParamExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-5)
    def test_r_execution_with_fixed_subset_size(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_cooling_house())
        ex.parameters['alpha'] = 0.05
        ex.parameters['verbose'] = 1
        ex.parameters['subset_size'] = 0
        db.session.commit()

        # When
        job, result = self.run_experiment(ex)

        # Then
        assert result.job_id == job.id
        assert result.job.experiment_id == ex.id
        assert result.start_time == job.start_time

        sepsets = db.session.query(Sepset).all()
        # If m.max=0, there can be no separation sets
        assert len(sepsets) == 0


class LogExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-6)
    def test_logs(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_gauss())
        ex.parameters['verbose'] = 1
        db.session.commit()

        # When
        job, result = self.run_experiment(ex)

        # Then
        assert result.job_id == job.id

        full_log = requests.get(self.url_for(JobLogsResource, job_id=job.id))
        assert full_log.status_code == 200
        assert 'Attaching package: ‘BiocGenerics’' in full_log.text
        assert 'Load dataset from' in full_log.text
        assert 'Successfully loaded dataset' in full_log.text
        assert 'Casting arguments...' in full_log.text
        assert 'Successfully executed job' in full_log.text

        offset_log = requests.get(self.url_for(JobLogsResource, job_id=job.id, offset=23))
        assert offset_log.status_code == 200
        assert 'Attaching package: ‘BiocGenerics’' not in offset_log.text
        assert 'Load dataset from' in offset_log.text
        assert 'Successfully loaded dataset' in offset_log.text
        assert 'Casting arguments...' in offset_log.text
        assert 'Successfully executed job' in offset_log.text

        limit_log = requests.get(self.url_for(JobLogsResource, job_id=job.id, limit=1))
        assert limit_log.status_code == 200
        assert 'Attaching package: ‘BiocGenerics’' not in limit_log.text
        assert 'Load dataset from' not in limit_log.text
        assert 'Successfully loaded dataset' not in limit_log.text
        assert 'Casting arguments...' not in limit_log.text
        assert 'Successfully executed job' in limit_log.text

        logfile = get_logfile_name(job.id)
        assert os.path.isfile(logfile)
        delete_request = requests.delete(self.url_for(JobLogsResource, job_id=job.id))
        assert delete_request.status_code == 200
        assert not os.path.isfile(logfile)
