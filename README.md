# Hierarchical Bradley-Terry Models with Dynamic Abilities

Project on Dynamic Hierarchical Bradley-Terry models for BMath(Hons) at UQ (2025-2026).

This repository primarily contains code and demos related to the project itself.
The key applications of the models was to analysing various home-ground advantage structures in the AFL and NRL.
See my [Honours Thesis](honours_thesis.pdf) for a comprehensive write-up of these findings (and more)!
Jupyter notebook files used for the analyses are included in the [demos](demos) folder.

I hope to produce a Python package with the models and tools for use in general analysis.
As such, certain aspects like external documentation, testing, and general polish are lacking at time of writing.

## Python

Python can be downloaded [here](https://www.python.org/downloads/).
It is recommended to be working in a virtual environment.
This is made pretty simple by running the following commands:

```bash
python -m venv env          # to init
.\env\Scripts\activate      # Windows
source env/bin/activate     # MacOS / Linux
deactivate                  # to exit
```

Jupyter may not recognise the virtual environment as a kernel.
I followed [this guide](https://web.archive.org/web/20240430135149/https://anbasile.github.io/posts/2017-06-25-jupyter-venv/) as a solution.
After restarting VSCode, `env` appears as a virtual environment.

```bash
pip install ipykernel
ipython kernel install --user --name=env
```

### Requirements

To install the required dependencies (preferably in your venv), run the following command:

```bash
pip install -r requirements.txt
```

Updating the requirements file is very simple.
The following command will generate a file with all the dependencies in the current environment:

```bash
pip freeze > requirements.txt
```

## Data

The demo Jupyter notebooks for AFL and NRL analysis require match data.
Refer to the respective demos for sources from which the data can be downloaded.
The auxiliary data on tenants has been compiled manually and is hence subject to potential errors.
In future, I may try to upload cleaned versions of the match datasets.
