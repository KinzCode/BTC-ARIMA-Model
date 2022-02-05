# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:17:56 2022

@author: KinzCode
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import pmdarima as pm
import json


def adf_test(target_series):
    """
    Parameters
    ----------
    target_series : Pandas Series
        The column of dataframe that contains the target data to which
        we want to test for stationarity e.g. Bitcoin Price.

    Returns
    -------
    adf_statistic : Float
        The adf statistic.
    p_value : Float
        The p value of the adf test.

    """
    result = adfuller(target_series)
    adf_statistic = result[0]
    p_value = result[1]
    print('ADF Statistic: %f' % adf_statistic)
    print('p-value: %f' % p_value)
    return adf_statistic, p_value


def find_order_of_differencing(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.

    Returns
    -------
    d : INT
        An integar representing the number of differences to get time
        series stationary. When P value is < 0.05 from augmented
        dicky fuller test function will return d.

    """
    # get the adf statistic and p values
    adf_statistic, p_value = adf_test(df['Price(USD)'])
    # p value needs to be < 0.05 for time series to be stationary
    if p_value > 0.05:
        # set number of differences to 0
        d = 0
        # if p value is above threshold repeatedly run logic until below
        while p_value > 0.05:
            print("")
            print("")
            print("P value to large, trying differencing")
            # difference the time series
            df['Price(USD)'] = df['Price(USD)'].diff()
            # drop the null values
            df.dropna(inplace = True)
            # add 1 to d for each iteration to represent 1 differencing
            d += 1
            # perform adf test again to asses p value and exit loop if stationary
            adf_statistic, p_value = adf_test(df['Price(USD)'])
        print(f"Success... TS now stationary after {d} differncing")
        return d
  
    
def create_acf_pacf(df):
    """
    Parameters
    ----------
    df : Data Frame
        The differenced dataframe generated from the find_order_differencing
        function.
    
    Function creates the auto correlation function and partial autocorrelation
    plots.
    Returns
    -------
    None.

    """
    # create acf plot
    acf_plot = plot_acf(df['Price(USD)'], lags=20)
    plt.show()
    # create pacf plot
    plot_pacf(df['Price(USD)'], lags = 20)
    plt.show()
    

def eda(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.
    
    Function finds d for p,d,q and creates acf/pacf plots
    Returns
    -------
    None.

    """
    d = find_order_of_differencing(df)
    create_acf_pacf(df)
    

def auto_arima(orig_df):
    """
    Parameters
    ----------
    orig_df : Data Frame
        Copied data frame from origonal read in prior to manual differencing.

    Returns
    -------
    model.order: Tuple
        tuple of found p,q,q values from auto arima

    """
    # get target series
    orig_df = orig_df['Price(USD)']
    # training set always roughly a months data
    train_length = len(orig_df) - 31
    train = orig_df[:train_length]
    model = pm.auto_arima(train,
                          start_p=1,
                          tart_q=1,test='adf',
                          max_p=15, max_q=15, 
                          m=1,
                          d=None,           
                          seasonal=False,   
                          start_P=0, 
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=False)
    
    # difference df by d found by auto arima
    differenced_by_auto_arima = orig_df.diff(model.order[1])
    return model.order, differenced_by_auto_arima


if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    # create df copy for auto arima that wont be differnce by eda process
    orig_df = df.copy()
    #perform eda - automate d and view plots for p and q
    eda(df)
    # Use auto arima to auto find p,d,q
    auto_p_d_q, differenced_by_auto_arima = auto_arima(orig_df)
    
    auto_p_d_q = {k:v for k, v in enumerate(auto_p_d_q)}
    
    # saved difference df to dat/clean for model
    df.to_csv(f'../dat/clean/differenced_{selected_currency}_daily_historical.csv')
    differenced_by_auto_arima.to_csv(f'../dat/clean/differenced_auto_arima_{selected_currency}_daily_historical.csv')
    
    # save auto p,d,q to json
    with open('auto_p_d_q.json', 'w') as fp:
        json.dump(auto_p_d_q, fp)
    
    