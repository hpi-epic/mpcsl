import signal
from unittest.mock import patch

from sqlalchemy import inspect

from src.master.resources.jobs import JobListResource, JobResource
from test.factories import JobFactory
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
        assert inspect(job).detached is True
