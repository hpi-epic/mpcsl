import numpy as np
import pandas as pd

from src.db import db
from src.master.resources import MarginalDistributionResource, ConditionalDistributionResource, \
    InterventionalDistributionResource
from src.master.resources.distributions import _custom_histogram
from test.factories import ResultFactory, NodeFactory, DatasetFactory, ExperimentFactory, JobFactory
from .base import BaseResourceTest


class MarginalDistributionTest(BaseResourceTest):

    def test_returns_continuous_marginal_distribution_for_node(self):
        # Given
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                "haha_.col" float,
                b float,
                c float
            );
        """)
        mean = [0, 5, 10]
        cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
        source = np.random.multivariate_normal(mean, cov, size=50)
        for l in source:
            db.session.execute("INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in l])))
        db.session.commit()

        exp = ExperimentFactory(dataset=ds)
        job = JobFactory(experiment=exp)
        result = ResultFactory(job=job)
        node = NodeFactory(dataset=result.job.experiment.dataset,
                           name='haha_.col')

        # When
        distribution = self.get(self.url_for(MarginalDistributionResource, node_id=node.id))

        # Then
        assert distribution['categorical'] is False
        assert distribution['node']['id'] == node.id
        assert distribution['dataset']['id'] == ds.id
        bins, bin_edges = _custom_histogram(source[:, 0], density=False)
        assert (distribution['bins'] == bins).all()
        assert np.allclose(distribution['bin_edges'], bin_edges)

    def test_returns_discrete_marginal_distribution_for_node(self):
        # Given
        ds = DatasetFactory(
            load_query="SELECT * FROM discrete_test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS discrete_test_data (
                "haha_.col" int,
                b int,
                c int
            );
        """)
        source = np.random.randint(5, size=(50, 3))
        for l in source:
            db.session.execute("INSERT INTO discrete_test_data VALUES ({0})".format(",".join([str(e) for e in l])))
        db.session.commit()

        exp = ExperimentFactory(dataset=ds)
        job = JobFactory(experiment=exp)
        result = ResultFactory(job=job)
        node = NodeFactory(dataset=result.job.experiment.dataset,
                           name='haha_.col')

        # When
        distribution = self.get(self.url_for(MarginalDistributionResource, node_id=node.id))

        # Then
        assert distribution['categorical'] is True
        assert distribution['node']['id'] == node.id
        assert distribution['dataset']['id'] == ds.id
        bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(source[:, 0], return_counts=True))])
        assert distribution['bins'] == bins
        assert 'bin_edges' not in distribution


class ConditionalDistributionTest(BaseResourceTest):

    def test_returns_discrete_conditional_distribution(self):
        # Given
        ds = DatasetFactory(
            load_query="SELECT * FROM cond_test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS cond_test_data (
                "haha_.col" int,
                b int,
                "Copy-haha_.col" int,
                "Copy-b" int
            );
        """)
        source = np.random.randint(2, size=(50, 2))
        source = np.concatenate((source, source + 1), axis=1)
        for l in source:
            db.session.execute("INSERT INTO cond_test_data VALUES ({0})".format(",".join([str(e) for e in l])))
        db.session.commit()

        exp = ExperimentFactory(dataset=ds)
        job = JobFactory(experiment=exp)
        ResultFactory(job=job)
        node = NodeFactory(dataset=job.experiment.dataset,
                           name='haha_.col')
        node2 = NodeFactory(dataset=job.experiment.dataset,
                            name='Copy-haha_.col')

        data = {
            'conditions': {
                node2.id: {
                    'values': [1],
                    'categorical': True
                }
            }
        }

        # When
        distribution = self.post(self.url_for(ConditionalDistributionResource, node_id=node.id), json=data)

        # Then
        assert distribution['categorical'] is True
        assert distribution['node']['id'] == node.id
        assert distribution['dataset']['id'] == ds.id

        conditioned_source = source[np.where(source[:, 2] == 1), 0]
        bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(conditioned_source, return_counts=True))])
        assert distribution['bins'] == bins
        assert 'bin_edges' not in distribution

    def test_auto_condition(self):
        # Given
        ds = DatasetFactory(
            load_query="SELECT * FROM cond_test_data2"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS cond_test_data2 (
                "haha_.col" int,
                b int,
                "haha_.floatcol" float
            );
        """)
        source = np.concatenate((np.random.randint(2, size=(50, 2)), np.random.normal(size=(50, 1))), axis=1)
        for l in source:
            db.session.execute("INSERT INTO cond_test_data2 VALUES ({0})".format(",".join([str(e) for e in l])))
        db.session.commit()

        values, counts = np.unique(source[:, 1], return_counts=True)
        cat_filter = values[np.argmax(counts)]
        hist, bin_edges = _custom_histogram(source[:, 2], density=False)
        most_common = np.argmax(hist)
        exp_distribution = source[:, 0][
            (source[:, 1] == cat_filter) &
            (source[:, 2] >= bin_edges[most_common]) &
            (source[:, 2] <= bin_edges[most_common+1])]
        exp_bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(exp_distribution, return_counts=True))])

        exp = ExperimentFactory(dataset=ds)
        job = JobFactory(experiment=exp)
        ResultFactory(job=job)
        node = NodeFactory(dataset=job.experiment.dataset,
                           name='haha_.col')
        node2 = NodeFactory(dataset=job.experiment.dataset,
                            name='b')
        node3 = NodeFactory(dataset=job.experiment.dataset,
                            name='haha_.floatcol')
        data = {
            'conditions': {
                node2.id: {
                    'auto': True,
                },
                node3.id: {
                    'auto': True,
                }
            }
        }

        # When
        distribution = self.post(self.url_for(ConditionalDistributionResource, node_id=node.id), json=data)

        assert distribution['categorical'] is True
        assert distribution['node']['id'] == node.id
        assert distribution['dataset']['id'] == ds.id
        assert distribution['bins'] == exp_bins
        assert 'bin_edges' not in distribution


class InterventionalDistributionTest(BaseResourceTest):

    def test_returns_interventional_distribution(self):
        # Given
        df = pd.read_csv('test/fixtures/coolinghouse_1k.csv', index_col=0)
        disc_df = df.apply(lambda c: pd.qcut(c, 8).cat.codes, axis=0)  # discretize
        disc_df.to_sql('test_data', con=db.engine, index=False)
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data",
            content_hash=None
        )
        db.session.commit()

        cause_node = NodeFactory(dataset=ds, name='V3')
        effect_node = NodeFactory(dataset=ds, name='V4')
        factor_nodes = [NodeFactory(dataset=ds, name='V2')]
        treatment = '7'
        data = {
            'cause_node_id': cause_node.id,
            'effect_node_id': effect_node.id,
            'factor_node_ids': [n.id for n in factor_nodes],
            'cause_condition': {
                'categorical': True,
                'values': [treatment]
            }
        }

        # When
        distribution = self.post(self.url_for(InterventionalDistributionResource), json=data)

        # Then
        assert 'node' in distribution
        assert distribution['node']['id'] == effect_node.id

        assert 'dataset' in distribution
        assert distribution['dataset']['id'] == ds.id

        assert 'bins' in distribution
        print(distribution['bins'])
        assert distribution['bins'] == {
            '3': 125,
            '5': 166,
            '4': 52,
            '0': 0,
            '6': 265,
            '2': 0,
            '7': 267,
            '1': 125
        }

    def test_noncategorical(self):
        # Given
        df = pd.read_csv('test/fixtures/coolinghouse_1k.csv', index_col=0)  # More than 10 categories
        df.to_sql('test_data2', con=db.engine, index=False)
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data2",
            content_hash=None
        )
        db.session.commit()

        cause_node = NodeFactory(dataset=ds, name='V4')
        effect_node = NodeFactory(dataset=ds, name='V2')
        factor_nodes = [NodeFactory(dataset=ds, name='V6')]
        data = {
            'cause_node_id': cause_node.id,
            'effect_node_id': effect_node.id,
            'factor_node_ids': [n.id for n in factor_nodes],
            'cause_condition': {
                'categorical': False,
                'from_value': 0.0,
                'to_value': 5.0
            }
        }

        # When
        distribution = self.post(self.url_for(InterventionalDistributionResource), json=data)

        # Then
        assert distribution['bins'] == [
            0, 0, 0, 0, 0, 0, 0, 4, 39, 153, 182, 78, 36, 5, 0, 0, 0, 0, 0, 503
        ]
        assert distribution['bin_edges'] == [
            -3.75535790044119, -3.367454309734735, -2.97955071902828, -2.591647128321825, -2.2037435376153702,
            -1.815839946908915, -1.4279363562024598, -1.0400327654960049, -0.6521291747895499, -0.26422558408309493,
            0.12367800662336004, 0.5115815973298155, 0.8994851880362704, 1.2873887787427254, 1.6752923694491804,
            2.0631959601556353, 2.4510995508620903, 2.8390031415685453, 3.2269067322750002, 3.614810322981455,
            4.00271391368791
        ]

    def test_empty_factors(self):
        # Given
        df = pd.read_csv('test/fixtures/coolinghouse_1k.csv', index_col=0)
        disc_df = df.apply(lambda c: pd.qcut(c, 8).cat.codes, axis=0)
        disc_df.to_sql('test_data3', con=db.engine, index=False)
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data3",
            content_hash=None
        )
        db.session.commit()

        cause_node = NodeFactory(dataset=ds, name='V3')
        effect_node = NodeFactory(dataset=ds, name='V4')
        factor_nodes = []
        treatment = '19'
        data = {
            'cause_node_id': cause_node.id,
            'effect_node_id': effect_node.id,
            'factor_node_ids': [n.id for n in factor_nodes],
            'cause_condition': {
                'categorical': True,
                'values': [treatment]
            }
        }

        # When
        distribution = self.post(self.url_for(InterventionalDistributionResource), json=data)

        # Then
        assert 'node' in distribution
        assert distribution['node']['id'] == effect_node.id

        assert 'dataset' in distribution
        assert distribution['dataset']['id'] == ds.id

        assert 'bins' in distribution
        assert distribution['bins'] == {
            '3': 0,
            '5': 0,
            '4': 0,
            '0': 0,
            '6': 0,
            '2': 0,
            '7': 0,
            '1': 1000
        }
