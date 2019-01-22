import pytest

from src.db import db
from src.models import Node, Edge, Sepset
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

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3


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

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3


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

        nodes = db.session.query(Node).all()
        for node in nodes:
            assert node.name in ['a', 'b', 'c']
        assert len(nodes) == 3


class SepsetExecutorTest(BaseIntegrationTest):

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
            assert (sepset.from_node.name, sepset.to_node.name, sepset.node_names) or \
                   (sepset.to_node.name, sepset.from_node.name, sepset.node_names) in sepset_set
        assert len(sepset_set) == len(sepsets)


class ParamExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-5)
    def test_r_execution_with_fixed_subset_size(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_cooling_house(), algorithm__script_filename='parallelpc.r')
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
