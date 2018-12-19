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

<img src="https://user-images.githubusercontent.com/8962207/50157097-d2869800-02d0-11e9-9c15-299442846712.png" width="600" title="FMC Architecture Diagram">

Additionally, the data model can be seen as ER diagram:

<img src="https://user-images.githubusercontent.com/8962207/50157111-e03c1d80-02d0-11e9-80a9-96d301355201.png" width="600" title="ER Datamodel Diagram">

## Setup

### Docker

The full backend setup is done using a docker container that means that you will require docker to be running for a local execution. Afterwards, you can clone the repository using:

```
git clone git@github.com:danthe96/mpci.git
```

Then we can move into the repository to build and execute the backend on a container using:

```
cp confdefault/backend.env conf/backend.env
docker-compose build
docker-compose up
```

This will start our backend with the default configuration.

### Setup with user interface
As the user interface files are store in a different repository (https://github.com/threxx/mpci-frontend),
they have to be initialized using:

```
git submodule init
git submodule update
```

When the UI files are present, the full setup can be build and started using

```
docker-compose -f docker-compose-nginx.yml build
docker-compose -f docker-compose-nginx.yml up
```

This will deploy the backend with an additional nginx server, that is used
to serve static files and provide the backend functionality by connecting to uWSGI.
The transpilation of the UI files will be done during build. If the UI files change,
it is necessary to rebuild the containers.

### Seeding the database
The files include a small seed-script that generates a randomized dataset.
The seed script can be run using:

```
docker-compose run backend python seed.py
```

If you are running the UI build, you have to include the -nginx.yml compose file.

### Migrating the database
As there are no migrations in place yet,
migrations are run using:
```
docker-compose down
docker-compose up
```

The first command will clear all volumes, including the database.

## Endpoint Documentation

A Swagger documentation of our REST endpoints is available using
http://localhost:5000/static/swagger/index.html
given default host and port settings.
