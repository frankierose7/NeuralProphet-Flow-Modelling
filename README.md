# NeuralProphet Flow Modelling


This repository contains jupyter notebooks (_.ipynb_ files) for exploratory data analysis and ARIMA, ARDL and NeuralProphet modelling, as well as a python script for optimisation of the NeuralProphet model. All require flow and rainfall datasets, accessible at https://environment.data.gov.uk/hydrology/explore and freely available under an Open Government Licence. (The files are too large to store in this repository.)

The scripts are designed to run with any flow dataset (specified in each script) and any rainfall datasets (specified in 'definitions.py')

The datasets used for analysis were:
- River Flow: Adelphi Weir
- Rainfall: Bacup, Blackstone Edge No 2, Bury, Cowm, Holden Wood, Kitcliffe, Loveclough, Ringley, Sweetloves
  
These datasets are stored in _.csv_ format, eg. `Adelphi-Weir-Upstream-Flow-15min-Qualified.csv`, and are generally around 200MB in size.

### 1. Exploratory Data Analysis
Exploratory data analysis on the flow and rainfall datasets is found in `eda.ipynb`.

Uses `eda_functions.py` which contains functions for loading the data and generating graphs, and `definitions.py` which contains definitions for rain gauge locations and data quality codes'

### 2. ARIMA and ARDL Models
ARIMA and ARDL models are found in `arima_ardl.ipynb`.

Uses `model_functions.py` to load data.

### 3. NeuralProphet model
The NeuralProphet model is found in `nprophet.ipynb`.
The optimised parameters for each horizon (1 hour, 3 hours and 6 hours) can be loaded from the three provided _.json_ files.

Uses `model_functions.py` to load data and set up the model.

### 4. Optimisation
The `nprophet_auto.py` script optimises the NeuralProphet model with the _hyperopt_ package. By default, it runs 75 iterations which takes around 5 hours. The result _.json_ files given are results of previous optimisations with this script.
