import factory
from sqlalchemy import inspect

from src.db import db
from src.master.resources.experiments import ExperimentListResource, ExperimentResource
from src.models import Experiment
from test.factories import ExperimentFactory, DatasetFactory, JobFactory, AlgorithmFactory
from .base import BaseResourceTest


class ExperimentTest(BaseResourceTest):
    def test_returns_all_experiments(self):
        # Given
        ex = ExperimentFactory()
        ex2 = ExperimentFactory()

        # When
        result = self.get(self.url_for(ExperimentListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == ex.id
        assert result[1]['id'] == ex2.id

    def test_returns_my_experiment(self):
        # Given
        ex = ExperimentFactory()
        job = JobFactory(experiment=ex)

        # When
        result = self.get(self.url_for(ExperimentResource, experiment_id=ex.id))

        # Then
        assert result['id'] == ex.id
        assert result['parameters']['alpha'] == ex.parameters['alpha']
        assert result['last_job']['id'] == job.id

    def test_create_new_experiment(self):
        # Given
        ds = DatasetFactory()
        alg = AlgorithmFactory()
        data = factory.build(dict, FACTORY_CLASS=ExperimentFactory)
        data.pop('dataset')
        data.pop('algorithm')
        data['algorithm_id'] = alg.id
        data['dataset_id'] = ds.id

        # When
        result = self.post(self.url_for(ExperimentListResource), json=data)
        ex = db.session.query(Experiment).first()

        # Then
        assert ex.dataset_id == ds.id
        assert result['parameters']['alpha'] == \
            ex.parameters['alpha'] == \
            data['parameters']['alpha']

    def test_delete_experiment(self):
        # Given
        ex = ExperimentFactory()

        # When
        result = self.delete(self.url_for(ExperimentResource, experiment_id=ex.id))

        # Then
        assert result['id'] == ex.id
        assert inspect(ex).detached is True
