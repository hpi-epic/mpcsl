from .base import BaseResourceTest
from src.models import JobStatus
from test.factories import ExperimentFactory
from src.master.resources.executor import ExecutorResource


class ExecutorTest(BaseResourceTest):

    def test_returns_all_algorithms(self):
        # Given
        ex = ExperimentFactory()
        ex.dataset.load_query = 'SELECT 5 AS col1_name'
        ex.dataset.content_hash = "cee5da8b8b5d8089f7cd7c9c85fac0e390ea2c4d6c7b" + \
            "04edad1a09796705a832017344e04d4aab4d0adcac636466dc70a9d952ae417254a1d877fbe99eb3b750"
        res = self.post(self.url_for(ExecutorResource, experiment_id=ex.id))
        res.raise_for_status()

        # Then
        assert ex.last_job.status == JobStatus.waiting
