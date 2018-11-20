from sqlalchemy import inspect
import factory

from src.db import db
from src.master.resources.datasets import DatasetListResource, DatasetResource
from src.models import Dataset
from test.factories import DatasetFactory
from .base import BaseResourceTest


class DatasetTest(BaseResourceTest):
    def test_returns_all_data_sets(self):
        # Given
        ds = DatasetFactory()
        ds2 = DatasetFactory()

        # When
        result = self.get(self.api.url_for(DatasetListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == ds.id
        assert result[1]['id'] == ds2.id

    def test_returns_my_data_set(self):
        # Given
        ds = DatasetFactory()

        # When
        result = self.get(self.api.url_for(DatasetResource, dataset_id=ds.id))

        # Then
        assert result['id'] == ds.id
        assert result['load_query'] == ds.load_query

    def test_delete_data_set(self):
        # Given
        ds = DatasetFactory()

        # When
        result = self.delete(self.api.url_for(DatasetResource, dataset_id=ds.id))

        # Then
        assert result['id'] == ds.id
        assert inspect(ds).detached is True

    def test_update_data_set(self):
        # Given
        ds = DatasetFactory()
        update_data = factory.build(dict, FACTORY_CLASS=DatasetFactory)

        # When
        result = self.put(self.api.url_for(DatasetResource, dataset_id=ds.id), json=update_data)

        # Then
        assert ds.load_query == update_data['load_query'] == result['load_query']

    def test_create_new_data_set(self):
        # Given
        data = factory.build(dict, FACTORY_CLASS=DatasetFactory)

        # When
        result = self.post(self.api.url_for(DatasetListResource), json=data)
        ds = db.session.query(Dataset).first()

        # Then
        assert ds.load_query == data['load_query'] == result['load_query']
