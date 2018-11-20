from src.master.resources.datasets import DatasetLoadResource, DatasetListResource, DatasetResource
from .base import BaseResourceTest


class DatasetTest(BaseResourceTest):
    def test_returns_a_data_set(self):
        # Given


        # When
        result = self.get(self.api.url_for(DatasetListResource))

        # Then
        assert result['hello'] == 'world!!!1'
