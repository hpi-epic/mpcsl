import factory
from sqlalchemy import inspect

from src.db import db
from src.master.resources.experiments import ExperimentListResource, ExperimentResource
from src.models import Experiment
from test.factories import ExperimentFactory, DatasetFactory, JobFactory, AlgorithmFactory, ResultFactory
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

    def test_avg_execution_time(self):
        # Given
        ex_wo_jobs = ExperimentFactory()
        ex_wo_results = ExperimentFactory()
        ex_w_results = ExperimentFactory()
        job = JobFactory(experiment=ex_wo_results)
        job2 = JobFactory(experiment=ex_w_results)
        result = ResultFactory(job=job2)
        result2 = ResultFactory(job=job2)
        result.execution_time = 2.0
        result2.execution_time = 3.0

        # Then
        assert ex_wo_jobs.avg_execution_time == 0.0
        assert ex_wo_results.avg_execution_time == 0.0
        assert ex_w_results.avg_execution_time == 2.5