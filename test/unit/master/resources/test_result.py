from datetime import datetime
import signal
from unittest.mock import patch


from src.db import db
from src.master.resources.results import ResultListResource, ResultResource
from test.factories import ResultFactory
from .base import BaseResourceTest


class JobTest(BaseResourceTest):
    def test_returns_all_results(self):
        # Given
        result = ResultFactory()
        result2 = ResultFactory()

        # When
        results = self.get(self.api.url_for(ResultListResource))

        # Then
        assert len(result) == 2
        assert results[0]['id'] == result.id
        assert results[1]['id'] == result2.id

    def test_returns_my_job(self):
        # Given
        result = ResultFactory()  # TODO: Finish result factory

        # When
        full_result = self.get(self.api.url_for(ResultResource, result_id=result.id))

        # Then
        # TODO: Check nodes, edges, sepset...

    def test_delete_job(self):
        # Given
        result = ResultFactory()

        # When
        deleted_result = self.delete(self.api.url_for(ResultResource, result_id=result.id))

        # Then
        assert deleted_result['id'] == result.id
