# Master Project: Causal Inference Pipeline

This repository contains the backend of a Causal Inference pipeline build during the Master Project 2018/19 at the HPI chair for Enterprise Platform and Integration Concepts. The pipeline currently includes the following features, all of which are accessible via a REST API:

- Store causal inference ready datasets into our backend
- Set up causal inference experiments for different causal inference algorithms in R with different hyperparameter settings and dataset choice
- Run the experiments as jobs directly in our backend
- Manage all currently running jobs on the backend
- Deliver the results and meta information of past experiments 
- Show distributions and perform interventions on results
- Annotate results with additional infromation
- Extend the pipeline with new algorithms in their own execution environments (e.g. C++)

The following image shows the holistic architecture as a FMC diagram:

<img src="https://user-images.githubusercontent.com/1437509/55067517-2fd47700-5080-11e9-9107-a7e0e28afa67.png" width="600" title="FMC Architecture Diagram">

Additionally, the data model can be seen as ER diagram:

<img src="https://user-images.githubusercontent.com/2228622/55068955-43351180-5083-11e9-9cc3-1f7d5ffcd83b.png" width="600" title="ER Datamodel Diagram">

## Setup

### Docker

The complete setup is done using [Docker](https://docs.docker.com/install/) that means that you will require docker to be running for a local execution.
As the user interface files are stored in a different [repository](https://github.com/hpi-epic/mpci-frontend),
you have to clone the repo using:

```
git clone --recurse-submodules git@github.com:hpi-epic/mpci.git
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

The example dataset for `demo.sh` is generated from an EARTHQUAKE bayesian network on [this page](http://www.bnlearn.com/bnrepository/discrete-small.html#earthquake).
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

## Contributors

* [Alexander Kastius](https://github.com/Raandom)
* [Victor Kuenstler](https://github.com/VictorKuenstler)
* [Tobias Nack](https://github.com/Dencrash)
* [Jonathan Schneider](https://github.com/jonaschn)
* [Daniel Thevessen](https://github.com/danthe96)
* [Theresa Zobel](https://github.com/threxx)
