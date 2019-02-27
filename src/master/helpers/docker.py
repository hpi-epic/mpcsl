import docker


def get_client():
    return docker.DockerClient(base_url='unix://var/run/docker.sock')
