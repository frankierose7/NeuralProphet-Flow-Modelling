# NeuralProphet Flow Modelling


This repository contains jupyter notebooks (_.ipynb_ files) for exploratory data analysis and ARIMA, ARDL and NeuralProphet modelling, as well as a python script for optimisation of the NeuralProphet model. All require flow and rainfall datasets, accessible at https://environment.data.gov.uk/hydrology/explore and freely available under an Open Government Licence. (The files are too large to store in this repository.)

The scripts are designed to run with any flow dataset (specified in each script) and any rainfall datasets (specified in _definitions.py_)

The datasets used for analysis were:
- River Flow: Adelphi Weir
- Rainfall: Bacup, Blackstone Edge No 2, Bury, Cowm, Holden Wood, Kitcliffe, Loveclough, Ringley, Sweetloves
These datasets are stored in _.csv_ format, eg. _Adelphi-Weir-Upstream-Flow-15min-Qualified.csv_

### 1. Exploratory Data Analysis
Exploratory data analysis on the flow and rainfall datasets is found in _eda.ipynb_.

Uses _eda_functions.py_ which contains functions for loading the data and generating graphs, and _definitions.py_ which contains definitions for rain gauge locations and data quality codes.

### 2. ARIMA and ARDL Models
ARIMA and ARDL models are found in _arima_ardl.ipynb_.

Uses _model_functions.py_ to load data.

### 3. NeuralProphet model
The NeuralProphet model is found in _nprophet.ipynb_.
The optimised parameters for each horizon (1 hour, 3 hours and 6 hours) can be loaded from the three provided _.json_ files.

Uses _model_functions.py_ to load data and set up the model.

### 4. Optimisation
The _nprophet_auto.py_ script optimises the NeuralProphet model with the _hyperopt_ package. By default runs 75 iterations which takes around 5 hours. The result _.json_ files given are results of previous optimisations with this script.
