import docker
from werkzeug.exceptions import Gone

from src.master.config import DOCKER_BASE_URL


def get_client():
    return docker.DockerClient(base_url=DOCKER_BASE_URL)


def get_container(container_id):
    try:
        client = get_client()
        return client.containers.get(container_id)
    except docker.errors.NotFound:
        raise Gone(f'Could not access container {container_id}. It probably does not exist anymore.')
