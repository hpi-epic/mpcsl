from datetime import datetime
import signal
from unittest.mock import patch


from src.db import db
from src.master.resources.jobs import JobListResource, JobResource, JobResultResource
from src.models import Experiment, Job, Result, Node, Edge
from test.factories import JobFactory, DatasetFactory
from .base import BaseResourceTest


class JobTest(BaseResourceTest):
    def test_returns_all_jobs(self):
        # Given
        job = JobFactory()
        job2 = JobFactory()

        # When
        result = self.get(self.api.url_for(JobListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == job.id
        assert result[1]['id'] == job2.id

    def test_returns_my_job(self):
        # Given
        job = JobFactory()

        # When
        result = self.get(self.api.url_for(JobResource, job_id=job.id))

        # Then
        assert result['id'] == job.id
        assert result['pid'] == job.pid

    def test_delete_job(self):
        # Given
        job = JobFactory()

        # When
        with patch('src.master.resources.jobs.os.kill') as m:
            result = self.delete(self.api.url_for(JobResource, job_id=job.id))

            m.assert_called_once_with(job.pid, signal.SIGTERM)

        # Then
        assert result['id'] == job.id
        assert result['status'] == "cancelled"

    def test_submit_results(self):
        # Given
        ds = DatasetFactory()
        mock_experiment = Experiment(dataset=ds)
        db.session.add(mock_experiment)
        mock_job = Job(experiment=mock_experiment, start_time=datetime.now())
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
        result = self.post(self.api.url_for(JobResultResource, job_id=mock_job.id), json=data)
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
