import numpy as np

from src.db import db
from src.master.resources import MarginalDistributionResource
from test.factories import ResultFactory, NodeFactory, DatasetFactory, ExperimentFactory, JobFactory
from .base import BaseResourceTest


class DistributionTest(BaseResourceTest):

    def test_returns_continuous_marginal_distribution_for_node(self):
        # Given
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )
        db.session.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                "MABT1_AS_137030ZE1_S7GC.AutoVR.aktiv..SK.in.Hand." float,
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
                           name='MABT1_AS_137030ZE1_S7GC.AutoVR.aktiv..SK.in.Hand.')

        # When
        distribution = self.get(self.url_for(MarginalDistributionResource, node_id=node.id))

        # Then
        assert distribution['categorical'] is False
        assert distribution['node']['id'] == node.id
        assert distribution['dataset']['id'] == ds.id
        bins, bin_edges = np.histogram(source[:, 0], bins='auto', density=False)
        assert (distribution['bins'] == bins).all()
        assert np.allclose(distribution['bin_edges'], bin_edges)

    def test_returns_discrete_marginal_distribution_for_node(self):
            # Given
            ds = DatasetFactory(
                load_query="SELECT * FROM discrete_test_data"
            )
            db.session.execute("""
                CREATE TABLE IF NOT EXISTS discrete_test_data (
                    "MABT1_AS_137030ZE1_S7GC.AutoVR.aktiv..SK.in.Hand." int,
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
                               name='MABT1_AS_137030ZE1_S7GC.AutoVR.aktiv..SK.in.Hand.')

            # When
            distribution = self.get(self.url_for(MarginalDistributionResource, node_id=node.id))

            # Then
            assert distribution['categorical'] is True
            assert distribution['node']['id'] == node.id
            assert distribution['dataset']['id'] == ds.id
            bins = dict([(str(k), int(v)) for k, v in zip(*np.unique(source[:, 0], return_counts=True))])
            assert distribution['bins'] == bins
            assert 'bin_edges' not in distribution
