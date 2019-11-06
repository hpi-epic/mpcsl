from src.master.appfactory import AppFactory
from flask_socketio import SocketIO

fact = AppFactory()

app = fact.up()
socketio = SocketIO(app)
db = fact.db

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port='5000', debug=True)
