from src.master.appfactory import AppFactory  # noqa: E402

fact = AppFactory()
app = fact.migration_up()
db = fact.db
