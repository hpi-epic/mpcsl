from .base import BaseResourceTest
from src.models import JobStatus
from test.factories import ExperimentFactory
from src.master.resources.executor import ExecutorResource


class ExecutorTest(BaseResourceTest):

    def test_returns_all_algorithms(self):
        # Given
        ex = ExperimentFactory()
        ex.dataset.load_query = 'SELECT 5 AS col1_name'
        ex.dataset.content_hash = "88720e21d69a08672cb77a30208de76347eac2e0b4da19ab7" \
            "eefea82259b9588e1fbc87505be5b27869b8d2b676f7276b7b0ab181b97d4e4babd56bc957f39bd"
        self.post(self.url_for(ExecutorResource, experiment_id=ex.id))

        # Then
        assert ex.last_job.status == JobStatus.waiting
