multi_modal_model

This repository provides a Dockerized environment for running a multi-modal model. Follow the instructions below to build and run the Docker image.

Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.


## Building the Docker Image

To build the Docker image, run the following command:

```bash
docker build -t multi_modal_model:latest .
```

This command builds a Docker image tagged as `multi_modal_model:latest` using the `Dockerfile` in the current directory.

## Running the Docker Container

Execute the following command to run the Docker container:

```bash
docker run --rm -it \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    multi_modal_model:latest
```
