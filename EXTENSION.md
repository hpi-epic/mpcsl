# Adding algorithms

The system was built to be easily extensible. This is achieved
by making use of docker images stored on the host system. Those 
images are used to execute the algorithm in their respective environments.

## Preparations

To add another algorithm to the system, the algorithm has to be available
as Docker image on the host system (or the execution system if configured).
Example Dockerfiles for the generation of such images are available in the 
executionenvironments directory in the source code of this project. The Dockerfile
should be built in a way that the following conditions are met:
- The entrypoint is set correctly so that the script name can be passed on launch 
  when running the image in a fresh container.
- The image contains all dependencies after build.
- The image already contains the script that calls the algorithm.

## Executing algorithms
The script or program that is executed in the container should be able to pursue
the following tasks:
1. Parse the command line parameters passed to it, that include the host of the API
2. Call the REST API to download the dataset
3. Execute the algorithm with the data given
4. Post the results back to the API to mark the job as done

All API endpoints are documented in Swagger.
For algorithms that are called or written in R, the mpciutils file in the
R environment can be reused to simplify the API calls. Examples can be found there
in the form of R scripts running PCAlg. The R image used by default for the R environment
contains two scripts, so two algorithms are defined later, that simply call the corresponding 
script using the same image.

### Command line parameters
The following command line parameters are passed to the script:
- **-j**: The job id (important for submitting results)
- **-d**: The dataset id (important when calling the API to load the dataset)
- **--api_host**: The host where the REST API is available
- **--send_sepsets**: Whether the REST API for result posting accepts separation sets in the results.
- All parameters specified in valid_parameters in the algorithm.json


## Adding the algorithm to the configuration
When the image containing the script file is prepared and built with a given
tag, the algorithm can be added to the algorithms.json file in the conf directory.
To add an algorithm there, add an object to the list with the parameters set correctly:
- name: Can be chosen freely, will be shown in the UI
- description: As above
- script_filename: The command to be run in the image to launch the algorithm.
- docker_image: The tag of the docker image that should be launched.
- valid_parameters: A dictionary (examples should be already there) that specifies
  additional valid command line parameters that the algorithm script accepts.
- docker_parameters: A dictionary containing additional arguments for the docker API 
  call. Available arguments can be found here: 
  https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run

When the algorithm is added to the file, restart the system to make it available 
for usage in experiments.