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

def adf_test(target_series):
    result = adfuller(target_series)
    adf_statistic = result[0]
    p_value = result[1]
    print('ADF Statistic: %f' % adf_statistic)
    print('p-value: %f' % p_value)
    return adf_statistic, p_value

def find_order_of_differencing(df):
    adf_statistic, p_value = adf_test(df['Price(USD)'])
    if p_value > 0.05:
        print("P value to large, trying first differnce")
        df['Price(USD)'] = np.log(df['Price(USD)'])
        df['Price(USD)'] = df['Price(USD)'].diff()
        df.dropna(inplace = True)
        adf_statistic, p_value = adf_test(df['Price(USD)'])
        if p_value < 0.05:
            print("Success... P value achieved after first differencing")
            return 1, df
        elif p_value > 0.50:
            print("Need to difference again")
    else:
        print("No difference Required")
    return 

def create_acf_plot(df):
    acf_plot = plot_acf(df['Price(USD)'], lags=100)
    print(acf_plot)
    return
    
def create_pacf_plot(df):
    plot_pacf(df['Price(USD)'], lags = 100)
    plt.show()
    
    
def find_p(df):
    create_acf_plot(df)
    create_pacf_plot(df)
    return

def eda(df):
    d, diff_df = find_order_of_differencing(df)
    p = find_p(diff_df)
    return d
    
def fit_arima(df):
    ts = df['Price(USD)']
    print("starting")
    model = sm.tsa.arima.ARIMA(ts, order = (2,1,2))
    model_fit = model.fit()
    print(model_fit.summary())
    plt.plot(ts)
    plt.plot(model_fit.fittedvalues, color = 'red', linewidth=0.5)
    model_fit.plot_predict(dynamic=False)
    plt.show()

def train_test_split(df):
    df = df['Price(USD)']
    df_length = len(df)
    
    train = df[:3165]
    test = df[3165:]
    # fit
    model = sm.tsa.arima.ARIMA(train, order = (2,2,2))
    fitted = model.fit()

    fc = fitted.get_forecast(38) 
    fc = (fc.summary_frame(alpha=0.0001))
    fc_mean = fc['mean']
    fc_lower = fc['mean_ci_lower']
    fc_upper = fc['mean_ci_upper']
  
    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(test, label='actual')
    plt.plot(fc_mean, label='mean_forecast', linewidth = 1.5)
    plt.plot(fc_lower, label = 'mean_ci_lower')
    plt.plot(fc_upper, label = 'mean_ci_upper')
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    model.plot_diagnostics(figsize=(15, 12))
    

def auto_arima(df):
    df = df['Price(USD)']
    train = df[:3165]
    test = df[3165:]
    model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=10, max_q=10, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=False)
    

    n_periods = 24
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    
    test.reset_index(drop = True, inplace = True)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(test, label='actual')
    plt.plot(fc, label='mean_forecast', linewidth = 1.5)
    plt.plot(confint[:,0], label = 'mean_ci_lower')
    plt.plot(confint[:,1], label = 'mean_ci_upper')
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    
    return fc, confint

if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    orig_df = df.copy()
    # perform eda - define P,D,Q
    #d = eda(df)
    p = 3
    q = 3
    # fit ARIMA
    #fit_arima(orig_df)
    #fc = train_test_split(orig_df)
    fc, confint = auto_arima(orig_df)