from src.master.appfactory import AppFactory

fact = AppFactory()

app = fact.up()
db = fact.db

if __name__ == "__main__":
    app.run(host="127.0.0.1", port='5000', debug=True)
