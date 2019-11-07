from src.master.config import MPCI_ENVIRONMENT
import eventlet
eventlet.sleep()
eventlet.monkey_patch()
from src.master.appfactory import AppFactory  # noqa: E402

fact = AppFactory()
[app, socketio] = fact.up()
db = fact.db


def main():
    isDevelopment = MPCI_ENVIRONMENT != 'production' and MPCI_ENVIRONMENT != 'staging'
    fact.start_daemon()
    socketio.run(app, host="0.0.0.0", port='5000', debug=isDevelopment)
    fact.stop_daemon()


if __name__ == "__main__":
    main()
