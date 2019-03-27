import docker
from werkzeug.exceptions import Gone

from src.master.config import DOCKER_BASE_URL, DOCKER_LOG_VOLUME_NAME


def get_client():
    return docker.DockerClient(base_url=DOCKER_BASE_URL)


def get_container(container_id):
    try:
        client = get_client()
        return client.containers.get(container_id)
    except docker.errors.NotFound:
        raise Gone(f'Could not access container {container_id}. It probably does not exist anymore.')


def get_or_create_log_volume():
    try:
        client = get_client()

        volumes = client.volumes.list()
        correct = list(filter(lambda o: o.name == DOCKER_LOG_VOLUME_NAME, volumes))

        if len(correct) == 0:
            correct = client.volumes.create(DOCKER_LOG_VOLUME_NAME)
        else:
            correct = correct[0]

        return correct

    except docker.errors.ApiError:
        raise Gone('Could not access Docker log volume.')
