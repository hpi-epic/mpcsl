def job_status_change(id, status):
    from src.app import socketio
    socketio.emit('job_status', {'id': id, 'status': status})
