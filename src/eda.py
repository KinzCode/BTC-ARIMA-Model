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
        d = 0
        while p_value > 0.05:
            print("")
            print("")
            print("")
            print("P value to large, trying first differnce")
            df['Price(USD)'] = df['Price(USD)'].diff()
            df.dropna(inplace = True)
            adf_statistic, p_value = adf_test(df['Price(USD)'])
            d += 1
        print(f"Success... TS now stationary after {d} differncing")
    
        
def create_acf_plot(df):
    acf_plot = plot_acf(df['Price(USD)'], lags=20)
    print(acf_plot)
    return
    
def create_pacf_plot(df):
    plot_pacf(df['Price(USD)'], lags = 20)
    plt.show()
      
def create_acf_pacf(df):
    create_acf_plot(df)
    create_pacf_plot(df)
    return

def eda(df):
    """
    Parameters
    ----------
    df : DataFrame
        cleaned time series data of the respective currency.
    Returns
    -------
    None.

    """
    find_order_of_differencing(df)
    create_acf_pacf(diff_df)
    
    

if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    #perform eda - define P,D,Q
    eda(df)
