import logging
import os
import yaml
from kubernetes import config, client
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
from kubernetes.client.rest import ApiException
from src.master.config import API_HOST, LOAD_SEPARATION_SET, RELEASE_NAME
from src.models import Job, Experiment

config.load_incluster_config()

api_instance = client.BatchV1Api()

JOB_PREFIX = f'{RELEASE_NAME}-execute-'


async def create_job(job: Job, experiment: Experiment):
    params = ['-j', str(job.id), '-d', str(experiment.dataset_id), '--api_host', str(API_HOST), '--send_sepsets', str(int(LOAD_SEPARATION_SET))]
    for k, v in experiment.parameters.items():
        params.append('--' + k)
        params.append(str(v))
    algorithm = experiment.algorithm
    command = ["/bin/sh",
      "-c","Rscript " + algorithm.script_filename + " " + " ".join(params)]
    with open(os.path.join(os.path.dirname(__file__), "executor-job.yaml")) as f:
        default_job = yaml.safe_load(f)
        job_name = f'{JOB_PREFIX}{job.id}'
        default_job["metadata"]["labels"]["job-name"] = job_name
        default_job["metadata"]["name"] = job_name
        default_job["spec"]["template"]["metadata"]["labels"]["job-name"] = job_name
        default_job["spec"]["template"]["spec"]["containers"][0]["command"] = command
        try:
            logging.info(f'Starting Job with ID {job.id}')
            result = api_instance.create_namespaced_job(namespace="default", body=default_job, pretty=True)
            return result.metadata.name
        except ApiException as e:
            logging.error("Exception when calling BatchV1Api->create_namespaced_job: %s\n" % e)


async def check_running_job(job: Job):
    result = api_instance.read_namespaced_job_status(job.container_id, "default")
    if result.status.failed is not None and result.status.failed > 0:
        deleteoptions = client.V1DeleteOptions()
        api_instance.delete_namespaced_job(job.container_id, "default", body=deleteoptions, grace_period_seconds=0, propagation_policy='Background')
        return True
    return False


def kube_delete_empty_pods(namespace='default'):
    deleteoptions = client.V1DeleteOptions()
    api_pods = client.CoreV1Api()
    try:
        pods = api_pods.list_namespaced_pod(namespace,
                                            pretty=True,
                                            timeout_seconds=60)
    except ApiException as e:
        logging.error("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

    for pod in pods.items:
        podname = pod.metadata.name
        is_execution_pod = podname.startswith(JOB_PREFIX)
        if is_execution_pod:
            try:
                if pod.status.phase == 'Succeeded' or pod.status.phase == 'Failed':
                    api_response = api_pods.delete_namespaced_pod(podname, namespace, body=deleteoptions)
                    logging.info("Pod: {} deleted!".format(podname))
                    logging.debug(api_response)
            except ApiException as e:
                logging.error("Exception when calling CoreV1Api->delete_namespaced_pod: %s\n" % e)
    return


def kube_cleanup_finished_jobs(namespace='default', state='Finished'):
    deleteoptions = client.V1DeleteOptions()
    try: 
        jobs = api_instance.list_namespaced_job(namespace,
                                                pretty=True,
                                                timeout_seconds=60)
    except ApiException as e:
        print("Exception when calling BatchV1Api->list_namespaced_job: %s\n" % e)
    for job in jobs.items:
        jobname = job.metadata.name

        if jobname.startswith(JOB_PREFIX) and job.status.succeeded == 1:
            # Clean up Job
            logging.info("Cleaning up Job: {}. Finished at: {}".format(jobname, job.status.completion_time))
            try: 
                # What is at work here. Setting Grace Period to 0 means delete ASAP. Otherwise it defaults to
                # some value I can't find anywhere. Propagation policy makes the Garbage cleaning Async
                api_response = api_instance.delete_namespaced_job(jobname,
                                                                  namespace,
                                                                  body=deleteoptions,
                                                                  grace_period_seconds=0,
                                                                  propagation_policy='Background')
                logging.debug(api_response)
            except ApiException as e:
                print("Exception when calling BatchV1Api->delete_namespaced_job: %s\n" % e)

    kube_delete_empty_pods(namespace)

    return


# if __name__ == "__main__":
    
    # print(config.list_kube_config_contexts())


# algorithm = experiment.algorithm

# params = [
#     '-j', str(new_job.id),
#     '-d', str(experiment.dataset_id),
#     '--api_host', str(API_HOST),
#     '--send_sepsets', str(int(LOAD_SEPARATION_SET))
# ]
# for k, v in experiment.parameters.items():
#     params.append('--' + k)
#     params.append(str(v))

# client = get_client()
# command = algorithm.script_filename + " " + " ".join(params)

# if DOCKER_MOUNT_LOG_VOLUME:
#     vol = get_or_create_log_volume()
#     volumes = {vol.name: {'bind': DOCKER_LOG_VOLUME_MOUNT_PATH, 'mode': 'rw'}}
#     params.append('--log_path')
#     params.append(DOCKER_LOG_VOLUME_MOUNT_PATH)
# else:
#     volumes = None

# try:
#     container = client.containers.run(
#         algorithm.docker_image,
#         command,
#         detach=True,
#         network=DOCKER_EXECUTION_NETWORK if DOCKER_EXECUTION_NETWORK else None,
#         volumes=volumes,
#         **algorithm.docker_parameters
#     )
# except docker.errors.ImageNotFound:
#     raise BadRequest(
#         f'Image {algorithm.docker_image} not found. Did you build '
#         f'the containers available in /src/executionenvironments?')

# new_job.container_id = container.id
# db.session.commit()
