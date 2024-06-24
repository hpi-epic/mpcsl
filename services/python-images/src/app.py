from src.master.config import MPCI_ENVIRONMENT
if MPCI_ENVIRONMENT == 'production' or MPCI_ENVIRONMENT == 'staging':
    import eventlet

    eventlet.sleep()
    eventlet.monkey_patch()
from src.master.appfactory import AppFactory  # noqa: E402

fact = AppFactory()
[app, socketio] = fact.up()
db = fact.db
ma = fact.ma
