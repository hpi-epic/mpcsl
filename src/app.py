from master.appfactory import AppFactory

fact = AppFactory()

app = fact.up()
db = fact.db
# print(app.url_map)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port='5000', debug=True)
