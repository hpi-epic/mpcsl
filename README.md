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

<img src="https://github.com/danthe96/mpci/blob/feature/readme/docs/fmc_architecture.png?raw=true" width="600" title="FMC Architecture Diagram">

Additionally, the data model can be seen as ER diagram:

<img src="https://github.com/danthe96/mpci/blob/feature/readme/docs/er_diagram.png" width="600" title="ER Datamodel Diagram">

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

## Endpoint Documentation

A Swagger documentation to our REST endpoints is available using
http://localhost:5000/static/ui/index.html
given default host and port settings.
