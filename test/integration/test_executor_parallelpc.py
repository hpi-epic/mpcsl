import pytest

from src.db import db
from src.models import Node, Edge, Sepset
from test.factories import ExperimentFactory
from .base import BaseIntegrationTest


class GaussExecutorTest(BaseIntegrationTest):

    @pytest.mark.run(order=-8)
    def test_r_execution_gauss(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_gauss(), algorithm__script_filename='parallelpc.r')
        ex.parameters['cores'] = 2
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

    @pytest.mark.run(order=-7)
    def test_r_execution_discrete(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_discrete(), algorithm__script_filename='parallelpc.r')
        ex.parameters['independence_test'] = 'disCI'
        ex.parameters['cores'] = 2
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

    @pytest.mark.run(order=-6)
    def test_r_execution_binary(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_binary(), algorithm__script_filename='parallelpc.r')
        ex.parameters['independence_test'] = 'binCI'
        ex.parameters['cores'] = 2
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

    @pytest.mark.run(order=-5)
    def test_r_execution_with_sepsets(self):
        # Given
        ex = ExperimentFactory(dataset=self.setup_dataset_cooling_house(), algorithm__script_filename='parallelpc.r')
        ex.parameters['alpha'] = 0.05
        ex.parameters['cores'] = 2
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
            assert (sepset.from_node.name, sepset.to_node.name, sepset.node_names) in sepset_set
        assert len(sepset_set) == len(sepsets)
