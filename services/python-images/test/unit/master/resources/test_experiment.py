import factory
from sqlalchemy import inspect

from src.db import db
from src.master.resources.experiments import ExperimentListResource, ExperimentResource
from src.models import Experiment
from test.factories import AlgorithmFactory, DatasetFactory, ExperimentFactory, ExperimentJobFactory, ResultFactory
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
        experiment_job = ExperimentJobFactory(experiment=ex)

        # When
        result = self.get(self.url_for(ExperimentResource, experiment_id=ex.id))

        # Then
        assert result['id'] == ex.id
        assert result['parameters']['alpha'] == ex.parameters['alpha']
        assert result['last_job']['id'] == experiment_job.id

    def test_change_experiment_description(self):
        ex = ExperimentFactory()
        ex.description = '1'
        result = self.put(self.url_for(ExperimentResource, experiment_id=ex.id),
                          json={'description': '2'})
        assert ex.description == '2'
        result = self.put(self.url_for(ExperimentResource, experiment_id=ex.id),
                          json={'sdgsdg': '3'},
                          parse_result=False)
        assert ex.description == '2'
        assert result.status_code == 400

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

    def test_execution_time_statistics(self):
        # Given
        ex_wo_jobs = ExperimentFactory()
        ex_wo_results = ExperimentFactory()
        ex_w_results = ExperimentFactory()
        ExperimentJobFactory(experiment=ex_wo_results)
        experiment_job2 = ExperimentJobFactory(experiment=ex_w_results)
        result = ResultFactory(job=experiment_job2)
        result2 = ResultFactory(job=experiment_job2)
        result.execution_time = 2.0
        result2.execution_time = 3.0

        # Then
        assert ex_wo_jobs.execution_time_statistics is None
        assert ex_wo_results.execution_time_statistics is None
        assert ex_w_results.execution_time_statistics['min'] == 2.0
        assert ex_w_results.execution_time_statistics['max'] == 3.0
        assert ex_w_results.execution_time_statistics['mean'] == 2.5
        assert ex_w_results.execution_time_statistics['median'] == 2.5
        assert ex_w_results.execution_time_statistics['lower_quantile'] == 2.25
        assert ex_w_results.execution_time_statistics['upper_quantile'] == 2.75
