from src.master.appfactory import AppFactory


fact = AppFactory()

[app, socketio] = fact.up()
db = fact.db

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port='5000', debug=True)
