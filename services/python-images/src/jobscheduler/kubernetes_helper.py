import logging
import os
import yaml
from kubernetes import config, client
from kubernetes.client.rest import ApiException
from src.master.config import API_HOST, LOAD_SEPARATION_SET, RELEASE_NAME, K8S_NAMESPACE, EXECUTION_IMAGE_NAMESPACE
from src.models import Job, Experiment, Algorithm

if os.environ.get("IN_CLUSTER") == "false":
    config.load_kube_config()
else:
    config.load_incluster_config()

api_instance = client.BatchV1Api()
core_api_instance = client.CoreV1Api()
JOB_PREFIX = f'{RELEASE_NAME}-execute-'


async def get_pod_log(job_id):
    try:
        pods: client.V1PodList = core_api_instance.list_namespaced_pod(namespace=K8S_NAMESPACE,
                                                                       label_selector=f'job-name=={JOB_PREFIX}{job_id}')
        if len(pods.items) == 0:
            return " -- EMPTY LOGS -- "
        pod: client.V1Pod = pods.items[0]
        if pod is None:
            return None
        pod_name = pod.metadata.name
        return core_api_instance.read_namespaced_pod_log(name=pod_name, namespace=K8S_NAMESPACE)
    except ApiException:
        logging.warning(f'No logs found for job {job_id}')
        return " -- EMPTY LOGS -- "


async def get_node_list():
    try:
        nodes: client.V1NodeList = core_api_instance.list_node()
        names = [node.metadata.labels["kubernetes.io/hostname"] for node in nodes.items]
        return names
    except ApiException as e:
        logging.error(e)
        return []


async def delete_job_and_pods(job_name):
    try:
        api_instance.delete_namespaced_job(job_name, namespace=K8S_NAMESPACE, propagation_policy='Background')
    except ApiException:
        logging.warning(f'Could not delete job {job_name}')
    try:
        core_api_instance.delete_collection_namespaced_pod(namespace=K8S_NAMESPACE,
                                                           label_selector=f'job-name=={job_name}',
                                                           propagation_policy='Background')
    except ApiException:
        logging.warning(f'Could not delete pods for job {job_name}')


async def create_job(job: Job, experiment: Experiment):
    params = ['-j', str(job.id),
              '-d', str(experiment.dataset_id),
              '--api_host', str(API_HOST),
              '--send_sepsets', str(int(LOAD_SEPARATION_SET))]
    for k, v in experiment.parameters.items():
        params.append('--' + k)
        params.append(str(v))
    algorithm: Algorithm = experiment.algorithm
    command = ["/bin/sh",
               "-c", "Rscript " + algorithm.script_filename + " " + " ".join(params)]
    with open(os.path.join(os.path.dirname(__file__), "executor-job.yaml")) as f:
        default_job = yaml.safe_load(f)
        job_name = f'{JOB_PREFIX}{job.id}'
        default_job["metadata"]["labels"]["job-name"] = job_name
        default_job["metadata"]["name"] = job_name
        default_job["spec"]["template"]["metadata"]["labels"]["job-name"] = job_name
        container = default_job["spec"]["template"]["spec"]["containers"][0]
        container["command"] = command
        container["image"] = \
            f'{EXECUTION_IMAGE_NAMESPACE}/{algorithm.docker_image}'
        if job.node_hostname is not None:
            nodeSelector = {
                "kubernetes.io/hostname": job.node_hostname
            }
            default_job["spec"]["template"]["spec"]["nodeSelector"] = nodeSelector
        if job.gpus is not None:
            container["resources"] = {
                "limits": {
                    "nvidia.com/gpu": str(job.gpus)
                }
            }
        try:
            logging.info(f'Starting Job with ID {job.id}')
            result = api_instance.create_namespaced_job(namespace=K8S_NAMESPACE, body=default_job, pretty=True)
            return result.metadata.name
        except ApiException as e:
            logging.error("Exception when calling BatchV1Api->create_namespaced_job: %s\n" % e)


async def check_running_job(job: Job):
    try:
        result = api_instance.read_namespaced_job_status(job.container_id, namespace=K8S_NAMESPACE)
        if result.status.failed is not None and result.status.failed > 0:
            deleteoptions = client.V1DeleteOptions()
            api_instance.delete_namespaced_job(job.container_id,
                                               namespace=K8S_NAMESPACE,
                                               body=deleteoptions,
                                               grace_period_seconds=0,
                                               propagation_policy='Background')
            return True
    except ApiException as e:
        if e.status == 404:
            return True
    return False


def kube_delete_empty_pods(session):
    deleteoptions = client.V1DeleteOptions()
    api_pods = client.CoreV1Api()
    try:
        pods = api_pods.list_namespaced_pod(namespace=K8S_NAMESPACE,
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
                    job_name = pod.metadata.labels["job-name"]
                    try:
                        logging.info(f'Saving logs for pod {podname}')
                        log = core_api_instance.read_namespaced_pod_log(name=podname, namespace=K8S_NAMESPACE)
                        job: Job = session.query(Job).filter(Job.container_id == job_name).one()
                        job.log = log
                        session.commit()
                    except ApiException:
                        logging.warning(f'No logs found for job {job_name}')
                    api_pods.delete_namespaced_pod(podname, namespace=K8S_NAMESPACE, body=deleteoptions)
                    logging.info("Pod: {} deleted".format(podname))
            except ApiException as e:
                logging.error("Exception when calling CoreV1Api->delete_namespaced_pod: %s\n" % e)
    return


def kube_cleanup_finished_jobs(session, state='Finished'):
    deleteoptions = client.V1DeleteOptions()
    try:
        jobs = api_instance.list_namespaced_job(namespace=K8S_NAMESPACE,
                                                pretty=True,
                                                timeout_seconds=60)
    except ApiException as e:
        print("Exception when calling BatchV1Api->list_namespaced_job: %s\n" % e)
    for job in jobs.items:
        jobname = job.metadata.name

        if jobname.startswith(JOB_PREFIX) and job.status.succeeded == 1:
            logging.info("Cleaning up Job: {}. Finished at: {}".format(jobname, job.status.completion_time))
            try:
                api_instance.delete_namespaced_job(jobname,
                                                   namespace=K8S_NAMESPACE,
                                                   body=deleteoptions,
                                                   grace_period_seconds=0,
                                                   propagation_policy='Background')
            except ApiException as e:
                print("Exception when calling BatchV1Api->delete_namespaced_job: %s\n" % e)

    kube_delete_empty_pods(session)

    return
