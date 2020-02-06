def job_status_change(id, error_code):
    from src.app import socketio
    socketio.emit('job', {'id': id, 'error_code': error_code})
