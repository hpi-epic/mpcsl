from src.app import socketio, app
from src.master.config import MPCI_ENVIRONMENT, PORT


if __name__ == "__main__":
    isDevelopment = MPCI_ENVIRONMENT != 'production' and MPCI_ENVIRONMENT != 'staging'
    port = PORT
    if port is None:
        port = 5000
    socketio.run(app, host="0.0.0.0", port=port, debug=isDevelopment)
