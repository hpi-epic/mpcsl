import factory
from sqlalchemy import inspect

from src.db import db
from src.master.resources.algorithms import AlgorithmResource, AlgorithmListResource
from src.models import Algorithm
from test.factories import AlgorithmFactory, DatasetFactory, JobFactory
from .base import BaseResourceTest


class AlgorithmTest(BaseResourceTest):
    def test_returns_all_algorithms(self):
        # Given
        alg = AlgorithmFactory()
        alg2 = AlgorithmFactory()

        # When
        result = self.get(self.url_for(AlgorithmListResource))

        # Then
        assert len(result) == 2
        assert result[0]['id'] == alg.id
        assert result[1]['id'] == alg2.id

    def test_create_new_algorithm(self):
        data = factory.build(dict, FACTORY_CLASS=AlgorithmFactory)
        print(data)

        # When
        result = self.post(self.url_for(AlgorithmListResource), json=data)
        ex = db.session.query(Algorithm).get(result['id'])

        # Then
        assert ex.id == result['id']

    def test_delete_algorithm(self):
        # Given
        ex = AlgorithmFactory()

        # When
        result = self.delete(self.url_for(AlgorithmResource,
                             algorithm_id=ex.id))

        # Then
        assert result['id'] == ex.id
        assert inspect(ex).detached is True
