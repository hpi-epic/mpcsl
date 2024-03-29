from datetime import datetime
from unittest.mock import patch, MagicMock

from src.db import db
from src.master.resources.jobs import JobListResource, JobResource, JobResultResource
from src.master.resources.experiment_jobs import ExperimentJobListResource
from src.models import Result, Edge, JobStatus, ExperimentJob
from test.factories import DatasetFactory, ExperimentFactory, ExperimentJobFactory, JobFactory, NodeFactory
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

    def test_returns_hidden_jobs(self):
        # Given
        job = JobFactory()
        job2 = JobFactory()
        job3 = JobFactory()
        job3.status = JobStatus.hidden

        # When
        result = self.get(self.url_for(JobListResource))
        result2 = self.get(self.url_for(JobListResource) + '?show_hidden=1')

        # Then
        assert len(result) == 2
        assert result[0]['id'] == job.id
        assert result[1]['id'] == job2.id

        assert len(result2) == 3
        assert result2[0]['id'] == job.id
        assert result2[1]['id'] == job2.id
        assert result2[2]['id'] == job3.id

    def test_returns_my_job(self):
        # Given
        job = JobFactory()

        # When
        result = self.get(self.url_for(JobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id
        assert result['container_id'] == job.container_id

    def test_returns_jobs_for_experiment(self):
        # Given
        job = ExperimentJobFactory()
        job2 = ExperimentJobFactory(experiment=job.experiment)
        if job2.start_time > job.start_time:
            j = job
            job = job2
            job2 = j

        ExperimentJobFactory()

        # When
        result = self.get(self.url_for(
            ExperimentJobListResource,
            experiment_id=job.experiment_id
        ))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == job.id
        assert result[1]['id'] == job2.id

    def test_cancel_job(self):
        # Given
        job = JobFactory()
        job.status = JobStatus.running

        # When
        m = MagicMock()
        with patch('src.master.resources.jobs.kill_container', m):
            result = self.delete(self.url_for(JobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id
        assert result['status'] == JobStatus.cancelled
        m.assert_called_once_with(job.container_id)

    def test_hide_job(self):
        # Given
        job = JobFactory()
        job.status = JobStatus.error

        # When
        result = self.delete(self.url_for(JobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id
        assert result['status'] == JobStatus.hidden

    def test_submit_results(self):
        # Given
        ds = DatasetFactory()
        mock_experiment = ExperimentFactory(dataset=ds)
        db.session.add(mock_experiment)
        mock_job = ExperimentJobFactory(experiment=mock_experiment, start_time=datetime.now())
        db.session.add(mock_job)
        nodes = [NodeFactory(name="X" + str(i + 1), dataset=ds) for i in range(3)]
        for node in nodes:
            db.session.add(node)
        db.session.commit()
        data = {
            'edge_list': [
                {'from_node': nodes[0].id, 'to_node': nodes[1].id, 'weight': 1.23}
            ],
            'sepset_list': [],
            'meta_results': {'important_note': 'lol', 'number': 123.123}
        }

        # When
        result = self.post(self.url_for(JobResultResource, job_id=mock_job.id), json=data)
        db_result: Result = db.session.query(Result).first()
        db_job: ExperimentJob = db.session.query(ExperimentJob).first()

        # Then
        assert db_job.end_time is not None
        assert db_result.meta_results == data['meta_results'] == result['meta_results']
        for edge in data['edge_list']:
            db_edge = db.session.query(Edge).filter(
                Edge.from_node.has(id=edge['from_node'])
            ).filter(
                Edge.to_node.has(id=edge['to_node'])
            ).first()
            assert db_edge is not None
            assert db_edge.weight == edge['weight']

    def test_submit_results_with_invalid_node_id(self):
        # Given
        ds = DatasetFactory()
        mock_experiment = ExperimentFactory(dataset=ds)
        db.session.add(mock_experiment)
        mock_job = ExperimentJobFactory(experiment=mock_experiment, start_time=datetime.now())
        db.session.add(mock_job)
        nodes = [NodeFactory(name="X" + str(i + 1), dataset=ds) for i in range(3)]
        for node in nodes:
            db.session.add(node)
        db.session.commit()
        data = {
            'edge_list': [
                {'from_node': nodes[0].id, 'to_node': -1, 'weight': 1.23}
            ],
            'sepset_list': [],
            'meta_results': {'important_note': 'lol', 'number': 123.123}
        }

        # When
        result = self.post(self.url_for(JobResultResource, job_id=mock_job.id), json=data, parse_result=False)

        # Then
        assert result.status_code == 400
