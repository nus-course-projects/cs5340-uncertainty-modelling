# Bayesian I3D for Word Level ASL

This repository hosts the bayesian augmentation for I3D to model uncertainity in word level ASL classification


## Getting Started

Create a virtual environment
```shell
python -m venv venv
```

Activate the virtual environment
- For Linux
```shell
source venv/bin/activate
```

- For Windows
```shell
.\venv\Scripts\activate
```

We use Poetry for dependency management as it let's us lock dependencies using a lockfile.
```shell
pip install poetry
```

Then, install the project dependencies
```shell
poetry install
```