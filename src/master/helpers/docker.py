import docker

from src.master.config import DOCKER_BASE_URL


def get_client():
    return docker.DockerClient(base_url=DOCKER_BASE_URL)
