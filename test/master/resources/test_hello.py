from src.master.resources.hello import HelloWorld
from .base import BaseResourceTest


class HelloTest(BaseResourceTest):
    def test_returns_hello_world(self):
        # When
        result = self.get(self.api.url_for(HelloWorld))

        # Then
        assert result['hello'] == 'world!!!1'
