import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet

from definitions import (locations, completeness_codes)
from functions import (get_flow_data, get_rainfall_data,
                       resample_flow_hr, resample_rf_hr)



def load_data(location):

    # load flow data for Adelphi Weir
    adelphi = get_flow_data(location)

    # load rainfall data for locations in the Irwell catchment
    rainfall_data = get_rainfall_data(locations)

    # group rainfall and flow data by hour
    adelphi_hr = resample_flow_hr(adelphi)
    rainfall_hr = resample_rf_hr(rainfall_data)

    # create mean rainfall dataframe
    rf_mean_hr = pd.DataFrame(index=rainfall_hr[locations[0]].index)
    for location in rainfall_hr.keys():
        rf_mean_hr[location] = rainfall_hr[location]['value']
    rf_mean_hr['rf_mean'] = rf_mean_hr[locations].mean(axis=1)
    rf_mean_hr = rf_mean_hr[['rf_mean']]

    # select flow data
    df_full = adelphi_hr
    df_full = df_full.rename(columns={'value':'y'})

    # select rainfall data
    df_full['rf_mean'] = rf_mean_hr[['rf_mean']]

    # offset rainfall data
    offset = 8
    df_full[f'rf_{offset}h'] = df_full['rf_mean'].shift(periods=offset)

    return df_full

def format_input(df):
    df['ds'] = df.index
    
    df.reset_index(drop=True, inplace=True)
    df = df[['ds', 'y',
             'rf_mean'
             ]]
    return df



def setup_model(params, n_forecasts):

    '''
    setup neural prophet model with parameters listed in 'params' dict
    '''

    model = NeuralProphet(
        # Trend
        growth='off',
        n_changepoints=0,
        # Seasonality
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        # Autoregression
        n_lags=params['ar_n_lags'], 
        ar_reg=params['ar_reg'], #default 0
        ar_layers=[params['ar_layer_size']]*params['ar_layer_num'],
        n_forecasts=n_forecasts, #steps ahead to forecast - default 1
        # Lagged Regressor
        lagged_reg_layers=[params['reg_layer_size']]*params['reg_layer_num'],
        # Uncertainty
        quantiles=[0.05,0.95],
        # Metrics
        collect_metrics={
            #'MSE': 'MeanSquaredError',
            #'R2': 'R2Score' # equivalent to Nash-Sutcliffe efficiency
            },  
        # Missing data
        impute_missing=True,
        impute_linear=10,
        impute_rolling=200,
        drop_missing=False
        )

    model.set_plotting_backend("plotly-static")

    model.add_lagged_regressor('rf_mean',
                    n_lags=params['reg_n_lags'], regularization=params['reg_reg'])
    
    return model