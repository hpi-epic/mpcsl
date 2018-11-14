from .appfactory import AppFactory

fact = AppFactory()

app = fact.up()
db = fact.db
