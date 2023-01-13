# Object Sensing and Cognition for Adversarial Robustness

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview
This repository contains the source code for Intel Lab's and Georgia Tech's
submissions to the DARPA GARD program. We currently distribute our submission
that contain our defenses, and we are working to open source our training code
too.

## Getting Started
To recreate our submission:
```
make submission
```

This assumes some dependencies are installed on your system. We support installing
these system-level dependencies for Ubuntu 18.04:
```
make ubuntu_deps
```

To run our submission, you will need to use [ARMORY](https://github.com/twosixlabs/armory).
We use `pyproject.toml` to manage our dependencies. You can install our project using:
```
pip install -e .
```

Because Armory primarily relies upon Docker images to run evaluations, we created our
own Docker image container our dependencies.
```
make docker_image
```

After install all Python dependencies, you need to configure Armory using:
```
armory configure
```

Once configured, you can run our defense submission via:
```
make run_submission
```

## Contributing
External contributions are not being accepted and pull requests will not be accepted.

## Acknowledgements
This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001119S0026.
