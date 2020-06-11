# Master Project: Causal Inference Pipeline
[![Docs](https://img.shields.io/badge/docs-wiki-blue.svg)](https://github.com/hpi-epic/mpci/wiki) [![CircleCI](https://circleci.com/gh/hpi-epic/mpci/tree/master.svg?style=svg&circle-token=a927c6324dcaf0d443e633300a3aa93d240c4193)](https://circleci.com/gh/hpi-epic/mpci/tree/master) [![codecov](https://codecov.io/gh/hpi-epic/mpci/branch/master/graph/badge.svg?token=64S6naWbgu)](https://codecov.io/gh/hpi-epic/mpci)

This repository contains the backend of a Causal Inference pipeline build during the Master Project 2018/19 at the HPI chair for Enterprise Platform and Integration Concepts. The pipeline currently includes the following features, all of which are accessible via a REST API:

- Store causal inference ready datasets into our backend
- Set up causal inference experiments for different causal inference algorithms in R with different hyperparameter settings and dataset choice
- Run the experiments as jobs directly in our backend
- Manage all currently running jobs on the backend
- Deliver the results and meta information of past experiments
- Show distributions and perform interventions on results
- Annotate results with additional infromation
- Extend the pipeline with new algorithms in their own execution environments (e.g. C++)

<!-- The following image shows the holistic architecture as a FMC diagram:

<img src="https://user-images.githubusercontent.com/1437509/55085207-92d90480-50a6-11e9-8f7e-e10fced882db.png" width="600" title="FMC Architecture Diagram">

Additionally, the data model can be seen as ER diagram:

<img src="https://user-images.githubusercontent.com/2228622/55068955-43351180-5083-11e9-9cc3-1f7d5ffcd83b.png" width="600" title="ER Datamodel Diagram"> -->

## Setup

### Requirements

- [Garden](https://github.com/garden-io/garden)
- [Minikube](https://github.com/kubernetes/minikube)

As the user interface files are stored in a different [repository](https://github.com/hpi-epic/mpci-frontend),
you have to clone the repo using:

```
git clone --recurse-submodules git@github.com:hpi-epic/mpci.git
```
### Getting Started

1. `minikube start`
2. `garden deploy`
3. `garden run task seed-db`
4. Goto `minikube ip` in browser

### Setup Algorithms

`garden run task db-setup-algorithms` loads the [algorithms](services/python-images/conf/algorithms.json) into the database.

### Seeding Example Dataset/Experiment

With `garden run task seed-db` an example dataset will be loaded into the database.
The example dataset is generated from an EARTHQUAKE bayesian network on [this page](http://www.bnlearn.com/bnrepository/discrete-small.html#earthquake).

## Endpoint Documentation

A Swagger documentation of our REST endpoints is available using `/swagger/index.html` given default host and port settings.

## Contributors

- [Marius Danner](https://github.com/MariusDanner)
- [Christopher Hagedorn](https://github.com/ChristopherSchmidt89)
- [Johannes Huegle](https://github.com/jh-hpi)
- [Alexander Kastius](https://github.com/Raandom)
- [Victor Kuenstler](https://github.com/VictorKuenstler)
- [Constantin Lange](https://github.com/constantin-lange)
- [Tobias Nack](https://github.com/Dencrash)
- [Milan Proell](https://github.com/milanpro)
- [Jonathan Schneider](https://github.com/jonaschn)
- [Daniel Thevessen](https://github.com/danthe96)
- [Theresa Zobel](https://github.com/threxx)
