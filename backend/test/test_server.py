
from src.server import hello


def test_hello_world():
    assert hello() == 'Hello World!'
