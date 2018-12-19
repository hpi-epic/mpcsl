import io

import numpy as np
import pandas as pd
import factory
from sqlalchemy import inspect
from marshmallow.utils import from_iso

from src.db import db
from src.master.resources.datasets import DatasetListResource, DatasetResource, DatasetLoadResource
from src.models import Dataset
from test.factories import DatasetFactory
from .base import BaseResourceTest


class DatasetTest(BaseResourceTest):
    def test_returns_all_data_sets(self):
        # Given
        ds = DatasetFactory()
        ds2 = DatasetFactory()

        # When
        result = self.get(self.url_for(DatasetListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == ds.id
        assert result[1]['id'] == ds2.id

    def test_returns_my_data_set(self):
        # Given
        ds = DatasetFactory()

        # When
        result = self.get(self.url_for(DatasetResource, dataset_id=ds.id))

        # Then
        assert result['id'] == ds.id
        assert result['load_query'] == ds.load_query
        assert result['name'] == ds.name
        assert result['description'] == ds.description
        assert from_iso(result['time_created']) == ds.time_created

    def test_create_new_data_set(self):
        # Given
        data = factory.build(dict, FACTORY_CLASS=DatasetFactory)

        # When
        result = self.post(self.url_for(DatasetListResource), json=data)
        ds = db.session.query(Dataset).first()

        # Then
        assert ds.load_query == data['load_query'] == result['load_query']

    def test_returns_the_correct_dataset(self):
        # Given
        ds = DatasetFactory(
            load_query="SELECT * FROM test_data"
        )

        db.session.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                a float,
                b float,
                c float
            );
        """)

        mean = [0, 5, 10]
        cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
        source = np.random.multivariate_normal(mean, cov, size=50)
        for l in source:
            db.session.execute("INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in l])))

        db.session.commit()

        # When
        result = self.test_client.get(self.url_for(DatasetLoadResource, dataset_id=ds.id))

        source = pd.DataFrame(source)
        source.columns = ['a', 'b', 'c']
        result = result.data.decode('utf-8')
        result = io.StringIO(result)
        result = pd.read_csv(result)

        # Then
        pd.testing.assert_frame_equal(source, result)

    def test_delete_dataset(self):
        # Given
        ex = DatasetFactory()

        # When
        result = self.delete(self.url_for(DatasetResource, dataset_id=ex.id))

        # Then
        assert result['id'] == ex.id
        assert inspect(ex).detached is True
