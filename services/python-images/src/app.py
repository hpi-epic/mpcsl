import eventlet

eventlet.sleep()
eventlet.monkey_patch()
from src.master.appfactory import AppFactory  # noqa: E402

fact = AppFactory()
[app, socketio] = fact.up()
db = fact.db
