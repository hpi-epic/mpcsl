import factory
from sqlalchemy import inspect

from src.db import db
from src.master.resources.algorithms import AlgorithmListResource, AlgorithmResource
from src.models import Algorithm
from test.factories import AlgorithmFactory
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

    def test_returns_my_algorithm(self):
        # Given
        alg = AlgorithmFactory()

        # When
        result = self.get(self.url_for(AlgorithmResource, algorithm_id=alg.id))

        # Then
        assert result['id'] == alg.id
        assert result['name'] == alg.name
        assert result['script_filename'] == alg.script_filename
        assert result['backend'] == alg.backend
        assert result['description'] == alg.description
        assert result['valid_parameters'] == alg.valid_parameters

    def test_create_new_algorithm(self):
        # Given
        data = factory.build(dict, FACTORY_CLASS=AlgorithmFactory)

        # When
        result = self.post(self.url_for(AlgorithmListResource), json=data)
        alg = db.session.query(Algorithm).get(result['id'])

        # Then
        assert alg.valid_parameters == data['valid_parameters'] == result['valid_parameters']

    def test_delete_algorithm(self):
        # Given
        alg = AlgorithmFactory()

        # When
        result = self.delete(self.url_for(AlgorithmResource, algorithm_id=alg.id))

        # Then
        assert result['id'] == alg.id
        assert inspect(alg).detached is True
