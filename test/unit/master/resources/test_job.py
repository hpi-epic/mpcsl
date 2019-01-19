from datetime import datetime
import signal
import factory
from unittest.mock import patch


from src.db import db
from src.master.resources.jobs import JobListResource, JobResource, JobResultResource, ExperimentJobListResource
from src.models import Result, Node, Edge
from test.factories import ExperimentFactory, JobFactory, DatasetFactory
from .base import BaseResourceTest


class JobTest(BaseResourceTest):
    def test_returns_all_jobs(self):
        # Given
        job = JobFactory()
        job2 = JobFactory()

        # When
        result = self.get(self.url_for(JobListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == job.id
        assert result[1]['id'] == job2.id

    def test_returns_my_job(self):
        # Given
        job = JobFactory()

        # When
        result = self.get(self.url_for(JobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id
        assert result['pid'] == job.pid

    def test_returns_jobs_for_experiment(self):
        # Given
        job = JobFactory()
        job2 = JobFactory(experiment=job.experiment)
        if job2.start_time > job.start_time:
            j = job
            job = job2
            job2 = j

        JobFactory()

        # When
        result = self.get(self.url_for(
            ExperimentJobListResource,
            experiment_id=job.experiment_id
        ))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == job.id
        assert result[1]['id'] == job2.id

    def test_delete_job(self):
        # Given
        job = JobFactory()
        job_pgid = factory.Faker('pyint')

        # When
        with patch('src.master.resources.jobs.os.getpgid', return_value=job_pgid) as mock_pgid:
            with patch('src.master.resources.jobs.os.killpg') as mock_killpg:
                result = self.delete(self.url_for(JobResource, job_id=job.id))

                mock_pgid.assert_called_once_with(job.pid)
                mock_killpg.assert_called_once_with(job_pgid, signal.SIGTERM)

        # Then
        assert result['id'] == job.id
        assert result['status'] == "cancelled"

    def test_submit_results(self):
        # Given
        ds = DatasetFactory()
        mock_experiment = ExperimentFactory(dataset=ds)
        db.session.add(mock_experiment)
        mock_job = JobFactory(experiment=mock_experiment, start_time=datetime.now())
        db.session.add(mock_job)
        db.session.commit()
        data = {
            'job_id': mock_job.id,
            'meta_results': {'important_note': 'lol'},
            'edge_list': [
                {'from_node': 'X1', 'to_node': 'X2'}
            ],
            'node_list': [
                'X1', 'X2', 'X3'
            ],
            'sepset_list': [
            ]
        }

        # When
        result = self.post(self.url_for(JobResultResource, job_id=mock_job.id), json=data)
        db_result = db.session.query(Result).first()

        # Then
        assert db_result.meta_results == data['meta_results'] == result['meta_results']
        for node in db.session.query(Node):
            assert node.name in data['node_list']
        for edge in data['edge_list']:
            assert db.session.query(Edge).filter(
                Edge.from_node.has(name=edge['from_node'])
            ).filter(
                Edge.to_node.has(name=edge['to_node'])
            ).first() is not None
