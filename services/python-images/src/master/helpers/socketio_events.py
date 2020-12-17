import os

from src.models import Job, JobErrorCode, ExperimentJob


def job_status_change(job: Job, error_code: JobErrorCode):
    from src.app import socketio
    socketio.emit('job', {'id': job.id, 'error_code': error_code})
    if isinstance(job, ExperimentJob):
        socketio.emit('experiment', {'id': job.experiment_id})


def dataset_node_change(dataset_id):
    if not os.getenv('TEST'):
        from src.app import socketio
        socketio.emit('dataset', {'id': dataset_id})
