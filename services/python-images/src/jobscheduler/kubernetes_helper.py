import logging
import asyncio
import os
import yaml
from kubernetes import config, client
from kubernetes.client.rest import ApiException
from src.master.config import API_HOST, LOAD_SEPARATION_SET, RELEASE_NAME, K8S_NAMESPACE, EXECUTION_IMAGE_NAMESPACE
from src.models import Job, ExperimentJob, DatasetGenerationJob, Algorithm, JobStatus, JobErrorCode
from src.jobscheduler.backend_requests import post_job_change

if os.environ.get("IN_CLUSTER") == "false":
    config.load_kube_config()
else:
    config.load_incluster_config()

api_instance = client.BatchV1Api()
core_api_instance = client.CoreV1Api()
JOB_PREFIX = f'{RELEASE_NAME}-execute-'

EMPTY_LOGS = " -- EMPTY LOGS -- "


async def get_pod_log(job_id):
    try:
        pods: client.V1PodList = core_api_instance.list_namespaced_pod(namespace=K8S_NAMESPACE,
                                                                       label_selector=f'job-name=={JOB_PREFIX}{job_id}')
        if len(pods.items) == 0:
            return EMPTY_LOGS
        pod: client.V1Pod = pods.items[0]
        if pod is None:
            return None
        pod_name = pod.metadata.name
        return core_api_instance.read_namespaced_pod_log(name=pod_name, namespace=K8S_NAMESPACE)
    except ApiException:
        logging.warning(f'No logs found for job {job_id}')
        return EMPTY_LOGS


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


async def create_experiment_job(experiment_job: ExperimentJob):
    experiment = experiment_job.experiment

    params = ['-j', str(experiment_job.id),
              '-d', str(experiment.dataset_id),
              '--api_host', str(API_HOST),
              '--send_sepsets', str(int(LOAD_SEPARATION_SET))]
    for k, v in experiment_job.experiment.parameters.items():
        params.append('--' + k)
        params.append(str(v))
    algorithm: Algorithm = experiment.algorithm
    subcommand = [algorithm.script_filename] + params
    with open(os.path.join(os.path.dirname(__file__), "executor-job.yaml")) as f:
        default_job = yaml.safe_load(f)
        job_name = f'{JOB_PREFIX}{experiment_job.id}'
        default_job["metadata"]["labels"]["job-name"] = job_name
        default_job["metadata"]["name"] = job_name
        default_job["spec"]["template"]["metadata"]["labels"]["job-name"] = job_name
        container = default_job["spec"]["template"]["spec"]["containers"][0]
        container["args"] = subcommand
        container["image"] = EXECUTION_IMAGE_NAMESPACE + "/" + algorithm.docker_image
        logging.info(container["image"])
        if experiment_job.node_hostname is not None:
            nodeSelector = {
                "kubernetes.io/hostname": experiment_job.node_hostname
            }
            default_job["spec"]["template"]["spec"]["nodeSelector"] = nodeSelector
        container["resources"] = {
                'limits': {},
                'requests': {}
            }
        if experiment_job.enforce_cpus and ('cores' in experiment.parameters):
            container["resources"]['limits']['cpu'] = experiment.parameters['cores']
            container["resources"]['requests']['cpu'] = experiment.parameters['cores']
        if experiment_job.gpus is not None:
            container["resources"]['limits']['nvidia.com/gpu'] = str(experiment_job.gpus)
            container["resources"]['requests']['nvidia.com/gpu'] = str(experiment_job.gpus)
        try:
            logging.info(f'Starting Job with ID {experiment_job.id}')
            logging.info(default_job)
            result = api_instance.create_namespaced_job(namespace=K8S_NAMESPACE, body=default_job, pretty=True)
            return result.metadata.name
        except ApiException as e:
            logging.error("Exception when calling BatchV1Api->create_namespaced_job: %s\n" % e)


async def create_dataset_generation_job(job: DatasetGenerationJob):
    params = [
        '--apiHost', API_HOST,
        '--uploadEndpoint', f'http://{API_HOST}/api/job/{job.id}/dataset_generation',
        '--nSamples', str(job.samples),
        '--nNodes', str(job.nodes),
        '--edgeProbability', str(job.edgeProbability),
        '--edgeValueLowerBound', str(job.edgeValueLowerBound),
        '--edgeValueUpperBound', str(job.edgeValueUpperBound)
    ]
    script_name = [
        "generator.r"
    ]
    subcommand = script_name + params
    docker_image = EXECUTION_IMAGE_NAMESPACE + "/generator_r"  # TODO Change this

    with open(os.path.join(os.path.dirname(__file__), "executor-job.yaml")) as f:
        default_job = yaml.safe_load(f)

        job_name = f'{JOB_PREFIX}{job.id}'
        default_job["metadata"]["labels"]["job-name"] = job_name
        default_job["metadata"]["name"] = job_name
        default_job["spec"]["template"]["metadata"]["labels"]["job-name"] = job_name
        container = default_job["spec"]["template"]["spec"]["containers"][0]
        container["args"] = subcommand
        container["image"] = docker_image

        logging.info(container["image"])

        if job.node_hostname is not None:
            nodeSelector = {
                "kubernetes.io/hostname": job.node_hostname
            }
            default_job["spec"]["template"]["spec"]["nodeSelector"] = nodeSelector
        container["resources"] = {
                'limits': {},
                'requests': {}
            }

        try:
            logging.info(f'Starting Job with ID {job.id}')
            logging.info(default_job)
            result = api_instance.create_namespaced_job(namespace=K8S_NAMESPACE, body=default_job, pretty=True)
            return result.metadata.name
        except ApiException as e:
            logging.error("Exception when calling BatchV1Api->create_namespaced_job: %s\n" % e)


def delete_job(jobname):
    deleteoptions = client.V1DeleteOptions()
    try:
        api_instance.delete_namespaced_job(jobname,
                                           namespace=K8S_NAMESPACE,
                                           body=deleteoptions,
                                           grace_period_seconds=0,
                                           propagation_policy='Background')
        logging.info("Job: {} deleted".format(jobname))
    except ApiException as e:
        print("Exception when calling BatchV1Api->delete_namespaced_job: %s\n" % e)


async def check_running_job(job: Job):
    try:
        result = api_instance.read_namespaced_job_status(job.container_id, namespace=K8S_NAMESPACE)
        if result.status.failed is not None and result.status.failed > 0:
            delete_job(job.container_id)
            return True
    except ApiException as e:
        if e.status == 404:
            return True
    return False


def get_job_for_pod(pod, session):
    job_name = pod.metadata.labels["job-name"]
    job: Job = session.query(Job).filter(Job.container_id == job_name).one()
    return job


def delete_pod(podname):
    deleteoptions = client.V1DeleteOptions()
    try:
        core_api_instance.delete_namespaced_pod(podname, namespace=K8S_NAMESPACE, body=deleteoptions)
        logging.info("Pod: {} deleted".format(podname))
    except ApiException as e:
        logging.error("Exception when calling CoreV1Api->delete_namespaced_pod: %s\n" % e)


async def handle_pending_pod(pod, session):
    last_condition = pod.status.conditions[-1]
    if last_condition is not None and last_condition.type == "PodScheduled":
        if last_condition.reason == "Unschedulable":
            job = get_job_for_pod(pod, session)
            job.status = JobStatus.error
            job.log = EMPTY_LOGS
            job.error_code = JobErrorCode.UNSCHEDULABLE
            session.commit()
            delete_job(job.container_id)
            delete_pod(pod.metadata.name)
            asyncio.create_task(post_job_change(job.id, job.error_code))
        elif last_condition.status == "True":
            cont_statuses = pod.status.container_statuses
            if cont_statuses is not None:
                last_cont_status = cont_statuses[-1]
                if last_cont_status.state is not None and last_cont_status.state.waiting is not None\
                        and last_cont_status.state.waiting.reason == "ImagePullBackOff":
                    job = get_job_for_pod(pod, session)
                    job.status = JobStatus.error
                    job.log = EMPTY_LOGS
                    job.error_code = JobErrorCode.IMAGE_NOT_FOUND
                    session.commit()
                    delete_job(job.container_id)
                    delete_pod(pod.metadata.name)
                    asyncio.create_task(post_job_change(job.id, job.error_code))


async def kube_delete_empty_pods(session):
    try:
        pods = core_api_instance.list_namespaced_pod(namespace=K8S_NAMESPACE,
                                                     pretty=True,
                                                     timeout_seconds=60)
    except ApiException as e:
        logging.error("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

    for pod in pods.items:
        podname = pod.metadata.name
        is_execution_pod = podname.startswith(JOB_PREFIX)
        if is_execution_pod:
            if pod.status.phase == 'Succeeded' or pod.status.phase == 'Failed':
                try:
                    logging.info(f'Saving logs for pod {podname}')
                    log = core_api_instance.read_namespaced_pod_log(name=podname, namespace=K8S_NAMESPACE)
                    job = get_job_for_pod(pod, session)
                    job.log = log
                    session.commit()
                except ApiException:
                    logging.warning(f'No logs found for job {podname}')
                delete_pod(podname)
            elif pod.status.phase == 'Pending':
                await handle_pending_pod(pod, session)
    return


async def kube_cleanup_finished_jobs(session):
    logging.info("--- Cleanup finished jobs and pods routine ---")
    try:
        jobs = api_instance.list_namespaced_job(namespace=K8S_NAMESPACE,
                                                pretty=True,
                                                timeout_seconds=60)
    except ApiException as e:
        print("Exception when calling BatchV1Api->list_namespaced_job: %s\n" % e)
    for job in jobs.items:
        job_name = job.metadata.name

        if job_name.startswith(JOB_PREFIX) and job.status.succeeded == 1:
            logging.info("Cleaning up Job: {}. Finished at: {}".format(job_name, job.status.completion_time))
            delete_job(job_name)
            asyncio.create_task(post_job_change(job.id, None))
    await kube_delete_empty_pods(session)
    return
