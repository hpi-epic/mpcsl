# Master Project: Causal Inference Pipeline

This repository contains the backend of a Causal Inference pipeline build during the Master Project 2018/19 at the HPI chair for Enterprise Platform and Integration Concepts. The pipeline currently includes the following features, which are all accessible via a REST-api:

- Store causal inference ready datasets into our backend
- Set up causal inference experiments for the pcalg algorithm in R with different hyperparameter settings and dataset choice
- Run the experiments as jobs directly in our backend
- Manage all currently running jobs on the backend
- Deliver the results and metainformation of past experiments 

The following features are currently under active development and will be added in the following months:

- Receive additional metainformation from past experiments
- Add the choice of additional causal inference algorithms
- Give people the opportunity to extend the pipeline with their own algorithms 
- Integrate prior knowledge into the algorithms
- Add additional steps to pre-process datasets

The following image shows the holistic architecture as a FMC diagram:

<img src="https://user-images.githubusercontent.com/1437509/55067517-2fd47700-5080-11e9-9107-a7e0e28afa67.png" width="600" title="FMC Architecture Diagram">

Additionally, the data model can be seen as ER diagram:

<img src="https://user-images.githubusercontent.com/8962207/50157111-e03c1d80-02d0-11e9-80a9-96d301355201.png" width="600" title="ER Datamodel Diagram">

## Setup

### Docker

The complete setup is done using [Docker](https://docs.docker.com/install/) that means that you will require docker to be running for a local execution.
As the user interface files are stored in a different [repository](https://github.com/VictorKuenstler/mpci-frontend),
you have to clone the repo using:

```
git clone --recurse-submodules git@github.com:danthe96/mpci.git
```

### [Scripts to rule them all](https://github.blog/2015-06-30-scripts-to-rule-them-all/)

To make your life easier to get from a git clone to an up-and-running project we prepared some scripts for you.
Here’s a quick mapping of what our scripts are named and what they’re responsible for doing:

- `bash scripts/bootstrap.sh` – installs/updates all dependencies
- `bash scripts/setup.sh` – sets up a project to be used for the first time
- `bash scripts/update.sh` – updates a project to run at its current version
- `bash scripts/server.sh` – starts app
- `bash scripts/demo.sh` – starts app with example dataset and experiment pre-configured
- `bash scripts/test.sh` – runs tests
- `bash scripts/cibuild.sh` – invoked by continuous integration servers to run tests
- `bash scripts/console.sh` – opens a console

Some of the scripts accept parameters that are passed through the underlying docker commands.
For example, you can start a server in detached mode with `bash scripts/server.sh --detach`
or run a specific test with `bash scripts/test.sh test/unit/master/resources/test_job.py`.

We provide three different ways how to run the Causal Inference Pipeline:

1. `backend` – starts just the backend with a postgres and a database ui - uses `docker-compose.yml`

1. `staging` – deploys the backend with an additional nginx server, that is used
to serve static files and provide the backend functionality by connecting to uWSGI.
The transpilation of the UI files will be done during build. - uses `docker-compose-staging.yml`

1. `production` – same setup as `staging` but without database ui. Make sure to override DB credentials - uses `docker-compose-prod.yml`

Change the environment variable `MPCI_ENVIRONMENT` in `conf/backend.env` accordingly to choose the desired setup.
The default is `backend`.

A database user interface is available using http://localhost:8081 given a `backend` or `staging` setup.

### Try it out

Just run `bash scripts/demo.sh` and open http://localhost:5000 in your browser.
Make sure to change the `MPCI_ENVIRONMENT` in your `conf/backend.env` to `staging` to also deploy the user interface.


## Endpoint Documentation

A Swagger documentation of our REST endpoints is available using
http://localhost:5000/static/swagger/index.html
given default host and port settings.

## Migrating the database


A clear database is needed to launch. This is especially important,
as the tests create all tables without using the migration system.
To get a clear database run:
```
bash scripts/setup.sh
```
This command will clear all volumes, including the database.

Otherwise you can run the following SQL command using the DB ui or a postgres 
admin interface of your choice.
```
DROP SCHEMA public CASCADE; CREATE SCHEMA public 
```

When the models have been changed, make sure your database is up to date by using:
```
bash scripts/update.sh
```

Afterwards, you can auto-create an migration by using:
```
docker-compose run --rm backend flask db migrate -m "migration message"
```
If alembic does not detect your changes correctly, you can manually create
an empty migration by using:
```
docker-compose run --rm backend flask db revision -m "migration message"
```

You can then either manipulate the autogenerated commands or insert new commands
in the new migration file.

When you are done, re-run upgrade to apply your changes to your local instance.

Alembic is used for the migration system. Alembic does not auto-detect the following changes correctly:
- Table and column renames (are detected as deleted and added with another name)
- Column type changes (are not detected at all, remember to add conversion function when adding them manually)

This list might not be complete, be sure to check the Alembic documentation for further information.

## FAQ

#### What should I do if following error occurs?
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server: Connection refused
	Is the server running on host "database" (192.168.64.2) and accepting
	TCP/IP connections on port 5432?
 (Background on this error at: http://sqlalche.me/e/e3q8)
```
For some reason the database startup took a bit too long. Just retry the command.
