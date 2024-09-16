
# WoWED-volcano
---

World Wide Experimental Design (WoWED) for volcano seismic monitoring is a collection of Jupiter notebooks that make the design of volcano seismic monitoring networks easy and intuitive. The notebooks and helper functions aim to allow fast and accuarate design of monitoring networks for volcanoes in an interactive way. A detailed description of the methods and the notebooks can be found in the following publication:
<!-- currently in progress -->
```
Strutz and Curtis, 2024, Near-real-time design of experiments for seismic monitoring of volcanoes, *in preparation*
```

## Getting Started

The easiest way of getting started is through our colab notebooks:

| Example | Link |
| --- | --- |
| basic interactive notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dominik-strutz/WoWED-volcano/blob/main/example_design_process.ipynb)    |


## Local Installation

To use the notebooks locally, you can clone the repository and install the required packages. The following commands will clone the repository and install the required packages:

### Using conda
```bash
git clone https://github.com/dominik-strutz/WoWED-volcano
cd WoWED-volcano
conda env create -f environment.yml
conda activate WoWED-volcano
```

### Using pip
```bash
git clone https://github.com/dominik-strutz/WoWED-volcano
cd WoWED-volcano
pip install -r requirements.txt
```

## Usage

After installing the required packages, you can start the Jupyter notebook server by running the following command:

```bash
jupyter notebook
```
or
```bash
jupyter lab
```

This will open a new tab in your browser where you can navigate to the notebooks and start using them.

