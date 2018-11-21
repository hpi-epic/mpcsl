import signal
from unittest.mock import patch

from sqlalchemy import inspect

from src.master.resources.jobs import JobListResource, JobResource
from test.factories import JobFactory
from .base import BaseResourceTest


class JobTest(BaseResourceTest):
    def test_returns_all_jobs(self):
        # Given
        ds = JobFactory()
        ds2 = JobFactory()

        # When
        result = self.get(self.api.url_for(JobListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == ds.id
        assert result[1]['id'] == ds2.id

    def test_returns_my_job(self):
        # Given
        ds = JobFactory()

        # When
        result = self.get(self.api.url_for(JobResource, job_id=ds.id))

        # Then
        assert result['id'] == ds.id
        assert result['load_query'] == ds.load_query

    def test_delete_job(self):
        # Given
        ds = JobFactory()

        # When
        with patch('src.master.resources.job.os') as m:
            result = self.delete(self.api.url_for(JobResource, job_id=ds.id))

            assert m.kill.assert_called_once_with(ds.pid, signal.SIGTERM)

        # Then
        assert result['id'] == ds.id
        assert inspect(ds).detached is True
