from .base import BaseResourceTest
from src.models import JobStatus
from test.factories import ExperimentFactory
from src.master.resources.executor import ExecutorResource


class ExecutorTest(BaseResourceTest):
    def test_returns_all_algorithms(self):
        # Given
        ex = ExperimentFactory()
        # When
        job = self.post(self.url_for(ExecutorResource, experiment_id=ex.id))

        # Then
        print(job)
        assert job['status'] == JobStatus.WAITING
