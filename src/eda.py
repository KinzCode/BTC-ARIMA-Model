# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:17:56 2022

@author: KinzCode
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
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


def kpss_test(target_series):
    print("Results of KPSS Test:")
    kpsstest = kpss(target_series, regression="ct", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

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
    # get kpss statiostic and p value
    kpss_test(df['Price(USD)'])
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
            # perform KPSS test
            kpss_test(df['Price(USD)'])
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
    # create acf
    fig, ax = plt.subplots(1, figsize=(12,8), dpi=100)
    plot_acf(df['Price(USD)'], lags=20, ax = ax)
    plt.ylim([-0.05, 0.25])
    plt.yticks(np.arange(-0.10,1.1, 0.1))
    plt.show()
    # create  pacf
    fig, ax = plt.subplots(1, figsize=(12,8), dpi=100)
    plot_pacf(df['Price(USD)'], lags = 20, ax = ax)
    plt.ylim([-0.05, 0.25])
    plt.yticks(np.arange(-0.10,1.1, 0.1))
    plt.show()

def describe_series(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.
    
    Function prints summary statistics of first half and second half of the time series.
    White noise will show a mean of 0 and a constant variance.

    Returns
    -------
    None.

    """
    first_half = int(len(df)/2)
    # Get summary for first half
    print(df['Price(USD)'][:first_half].describe())
    # get summary for second half
    print(df['Price(USD)'][first_half:].describe())

def histogram_series(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.
    
    Function creates histogram of first half and second half of the time series.
    White noise will have an identical distribution of values throughout the series.

    Returns
    -------
    None.

    """  
    first_half = int(len(df)/2)
    fig, ax = plt.subplots(1, figsize=(12,8), dpi=100)
    df['Price(USD)'][first_half:].hist()
    df['Price(USD)'][:first_half].hist()

def ljung_box_test(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.
    
    Function runs ljung box test. If value is < 0.05 null hypothesis can be
    rejected and the time series is unlikely white noise.
    Returns
    -------
    None.

    """      
    sm.stats.acorr_ljungbox(df['Price(USD)'], lags=[20], return_df=True)
    
def check_white_noise(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.
    
    Function is place holder for functions that test for white noise.

    Returns
    -------
    None.

    """
    describe_series(df)
    histogram_series(df)
    ljung_box_test(df)

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
    check_white_noise(df)


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
    differenced_by_auto_arima: Data Frame
        the differenced data frame output from the auto arima function
    fitted_residuals: The residuals of the fitted model.

    """
    #get target series
    orig_df = np.log(orig_df['Price(USD)'])
    model = pm.auto_arima(orig_df,
                          start_p=10,
                          start_q=10,
                          test='adf',
                          max_p=10, 
                          max_q=10, 
                          m=1,
                          d=None,           
                          seasonal=False,   
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True,
                         stepwise = True)
    # difference df by d found by auto arima
    differenced_by_auto_arima = orig_df.diff(model.order[1])
    return model.order, differenced_by_auto_arima, model.resid()


if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    # create df copy for auto arima that wont be differnce by eda process
    orig_df = df.copy()
    #perform eda - automate d and view plots for p and q
    eda(df)
    # Use auto arima to auto find p,d,q
    auto_p_d_q, differenced_by_auto_arima, fitted_residuals = auto_arima(orig_df)
    
    auto_p_d_q = {k:v for k, v in enumerate(auto_p_d_q)}
    
    # saved difference df to dat/clean for model
    df.to_csv(f'../dat/clean/differenced_{selected_currency}_daily_historical.csv')
    differenced_by_auto_arima.to_csv(f'../dat/clean/differenced_auto_arima_{selected_currency}_daily_historical.csv')
    
    # save auto p,d,q to json
    with open('auto_p_d_q.json', 'w') as fp:
        json.dump(auto_p_d_q, fp)
    
    