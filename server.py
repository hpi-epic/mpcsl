from src.app import socketio, app

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port='5000', debug=True)
