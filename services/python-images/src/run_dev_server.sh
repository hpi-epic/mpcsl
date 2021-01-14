# before running this you should set up a virtualenv and install the requirements.txt
export POSTGRES_USER=admin
export POSTGRES_PASSWORD=admin
export FLASK_APP=app.py
export FLASK_ENV=development
export SCHEDULER_HOST=localhost:4000
flask run --host 0.0.0.0
