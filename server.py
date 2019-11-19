from src.app import socketio, app
from src.master.config import MPCI_ENVIRONMENT


if __name__ == "__main__":
    isDevelopment = MPCI_ENVIRONMENT != 'production' and MPCI_ENVIRONMENT != 'staging'
    socketio.run(app, host="0.0.0.0", port='5000', debug=isDevelopment)
