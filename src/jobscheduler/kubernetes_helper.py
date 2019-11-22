import asyncio
import logging
import os
from kubernetes import config, client
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
from kubernetes.client.rest import ApiException
from src.master.config import DAEMON_CYCLE_TIME, SQLALCHEMY_DATABASE_URI, API_HOST, LOAD_SEPARATION_SET
from src.models import Job, JobStatus, Experiment


# async def watchWaitingJobs():
#     logging.info("Watching for waiting Jobs")
#     config.load_incluster_config()
#     engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
#     Session = sessionmaker(bind=engine)
#     while True:
#         await asyncio.sleep(DAEMON_CYCLE_TIME)
#         session = Session()
#         jobs = session.query(Job).filter(Job.status == JobStatus.running)
#         for job in jobs:
#             if job.status == JobStatus.waiting:

config.load_incluster_config()

api_instance = client.BatchV1Api()


async def createJob(job: Job, experiment: Experiment):
    params = ['-j', str(job.id), '-d', str(experiment.dataset_id), '--api_host', str(API_HOST), '--send_sepsets', str(int(LOAD_SEPARATION_SET))]
    for k, v in experiment.parameters.items():
        params.append('--' + k)
        params.append(str(v))
    algorithm = experiment.algorithm
    command = algorithm.script_filename + " " + " ".join(params)
    algorithm.docker_image
    # ns = os.getenv("K8S_NAMESPACE")
    # if ns is None:
    ns = "default"
    k8s_job_metadata = client.V1ObjectMeta(name="execution-job-"+str(job.id), namespace=ns)
    k8s_job_template_containers = [client.V1Container(name="execution-container", image="milanpro/mpci_executor", command=[command])]
    k8s_job_template_spec = client.V1PodSpec(containers=k8s_job_template_containers, restart_policy="Never")
    k8s_job_template = client.V1PodTemplateSpec(spec=k8s_job_template_spec)
    k8s_job_spec = client.V1JobSpec(template=k8s_job_template)
    k8s_job = client.V1Job(api_version="batch/v1", kind="Job", metadata=k8s_job_metadata, spec=k8s_job_spec)

    try:
        logging.info("Starting Job with ID %s", str(job.id))
        api_response = api_instance.create_namespaced_job(namespace=ns, body=k8s_job, pretty=True)
        print(api_response)
    except ApiException as e:
        print("Exception when calling BatchV1Api->create_namespaced_job: %s\n" % e)


def kube_delete_empty_pods(namespace='default', phase='Succeeded'):
    """
    Pods are never empty, just completed the lifecycle.
    As such they can be deleted.
    Pods can be without any running container in 2 states:
    Succeeded and Failed. This call doesn't terminate Failed pods by default.
    """
    # The always needed object
    deleteoptions = client.V1DeleteOptions()
    # We need the api entry point for pods
    api_pods = client.CoreV1Api()
    # List the pods
    try:
        pods = api_pods.list_namespaced_pod(namespace,
                                            include_uninitialized=False,
                                            pretty=True,
                                            timeout_seconds=60)
    except ApiException as e:
        logging.error("Exception when calling CoreV1Api->list_namespaced_pod: %s\n" % e)

    for pod in pods.items:
        logging.debug(pod)
        podname = pod.metadata.name
        try:
            if pod.status.phase == phase:
                api_response = api_pods.delete_namespaced_pod(podname, namespace, deleteoptions)
                logging.info("Pod: {} deleted!".format(podname))
                logging.debug(api_response)
            else:
                logging.info("Pod: {} still not done... Phase: {}".format(podname, pod.status.phase))
        except ApiException as e:
            logging.error("Exception when calling CoreV1Api->delete_namespaced_pod: %s\n" % e)

    return


def kube_cleanup_finished_jobs(namespace='default', state='Finished'):
    """
    Since the TTL flag (ttl_seconds_after_finished) is still in alpha (Kubernetes 1.12) jobs need to be cleanup manually
    As such this method checks for existing Finished Jobs and deletes them.
    By default it only cleans Finished jobs. Failed jobs require manual intervention or a second call to this function.
    Docs: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#clean-up-finished-jobs-automatically
    For deletion you need a new object type! V1DeleteOptions! But you can have it empty!
    CAUTION: Pods are not deleted at the moment. They are set to not running, but will count for your autoscaling limit, so if
             pods are not deleted, the cluster can hit the autoscaling limit even with free, idling pods.
             To delete pods, at this moment the best choice is to use the kubectl tool
             ex: kubectl delete jobs/JOBNAME.
             But! If you already deleted the job via this API call, you now need to delete the Pod using Kubectl:
             ex: kubectl delete pods/PODNAME
    """
    deleteoptions = client.V1DeleteOptions()
    try: 
        jobs = api_instance.list_namespaced_job(namespace,
                                                include_uninitialized=False,
                                                pretty=True,
                                                timeout_seconds=60)
        #print(jobs)
    except ApiException as e:
        print("Exception when calling BatchV1Api->list_namespaced_job: %s\n" % e)
    
    # Now we have all the jobs, lets clean up
    # We are also logging the jobs we didn't clean up because they either failed or are still running
    for job in jobs.items:
        logging.debug(job)
        jobname = job.metadata.name
        jobstatus = job.status.conditions
        if job.status.succeeded == 1:
            # Clean up Job
            logging.info("Cleaning up Job: {}. Finished at: {}".format(jobname, job.status.completion_time))
            try: 
                # What is at work here. Setting Grace Period to 0 means delete ASAP. Otherwise it defaults to
                # some value I can't find anywhere. Propagation policy makes the Garbage cleaning Async
                api_response = api_instance.delete_namespaced_job(jobname,
                                                                  namespace,
                                                                  deleteoptions,
                                                                  grace_period_seconds= 0, 
                                                                  propagation_policy='Background')
                logging.debug(api_response)
            except ApiException as e:
                print("Exception when calling BatchV1Api->delete_namespaced_job: %s\n" % e)
        else:
            if jobstatus is None and job.status.active == 1:
                jobstatus = 'active'
            logging.info("Job: {} not cleaned up. Current status: {}".format(jobname, jobstatus))
    
    # Now that we have the jobs cleaned, let's clean the pods
    kube_delete_empty_pods(namespace)
    # And we are done!
    return


if __name__ == "__main__":
    ns = os.getenv("K8S_NAMESPACE")
    if ns is None:
        ns = ""
    config.load_kube_config()
    api = client.CoreV1Api()
    jobs = api.list_namespaced_pod(namespace=ns, watch=False)
    for job in jobs.items:
        print("%-16s" %
              (job.metadata.name))
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
