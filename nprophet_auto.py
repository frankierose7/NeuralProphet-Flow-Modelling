#%%
'''
Script for running automatic optimisation of the neuralprophet model
'''
# file name to save optimisation result
file_name = 'np_hyperopt_result_3h.json'
# horizon to optimise in hours
HORIZON = 3

## LOAD PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
from sklearn.metrics import r2_score, mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from np_functions import load_data, format_input, setup_model


## SET UP TRAINING AND TEST DATASETS

df_full = load_data()

# small dataset
'''df_train = df_full.loc['2001'].copy()
df_test = df_full.loc['2002-01': '2002-02'].copy()'''

# large dataset
df_train = df_full.loc['2001':'2014'].copy()
#df_train = df_train[df_train['quality_min'] != 0] # remove missing data
df_test = df_full.loc['2015': '2019'].copy()

# training data
df_train = format_input(df_train)
# test data
df_test = format_input(df_test)


## AUTOMATIC MODEL OPTIMISATION


def objective(params, horizon=HORIZON):
    '''objective function to optimise'''

    # convert params to int where required
    for key, value in params.items():
        if key not in ['ar_reg', 'reg_reg']:
            params[key] = int(value)

    # setup model
    model = setup_model(params, n_forecasts = horizon)
    # fit model
    metrics_train = model.fit(df_train)
    # make predictions
    forecast = model.predict(df_test)

    # remove rows with missing predictions
    if horizon = 1:
            forecast = forecast.dropna(subset=['yhat1'])
    else:
        forecast = forecast.dropna(subset=['yhat1', f'yhat{horizon}'])

    # get prediction values
    y = forecast['y']

    cols = []
    for i in range(horizon-1):
        cols.append(f'yhat{i-1}')
    pred_cols = forecast[cols]

    ypreds = [forecast[col] for col in pred_cols]


    # calculate NSE
    nse_vals = [r2_score(ypred, y) for ypred in ypreds]

    loss = 1-nse_vals[horizon-1]

    return {'loss': loss,
        'nse_vals': nse_vals,
        'status': STATUS_OK
        }

space = {
        'yearly_seasonality': hp.choice('yearly_seasonality', [True, False]),
        'ar_n_lags': hp.quniform('ar_n_lags', 1, 720, 1),
        'ar_reg': hp.loguniform('ar_reg', -1, 2),
        'ar_layer_size': hp.quniform('ar_layer_size', 1, 100, 1),
        'ar_layer_num': hp.quniform('ar_layer_num', 1, 4, 1),
        'reg_n_lags': hp.quniform('reg_n_lags', 1, 720, 1),
        'reg_reg': hp.loguniform('reg_reg', -1, 2),
        'reg_layer_size': hp.quniform('reg_layer_size', 1, 100, 1),
        'reg_layer_num': hp.quniform('reg_layer_num', 1, 4, 1),
    }
trials = Trials()


best = fmin(objective,
    space = space,
    algo = tpe.suggest,
    max_evals = 1,
    trials = trials
)

## SAVE RESULTS TO TEXT FILE

# Function to convert numpy types to standard Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Save to a text file
with open(file_name, 'w') as file:
    json.dump({"best": best, "trials": trials.trials}, file, default=convert_numpy_types, indent=4)