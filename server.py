from src.app import socketio, app, fact
from src.master.config import MPCI_ENVIRONMENT

if __name__ == "__main__":
    isDevelopment = MPCI_ENVIRONMENT != 'production' and MPCI_ENVIRONMENT != 'staging'
    fact.start_daemon()
    socketio.run(app, host="0.0.0.0", port='5000', debug=isDevelopment)
    fact.stop_daemon()
