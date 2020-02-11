def job_status_change(job, error_code):
    from src.app import socketio
    socketio.emit('job', {'id': job.id, 'error_code': error_code})
    socketio.emit('experiment', {'id': job.experiment_id})
