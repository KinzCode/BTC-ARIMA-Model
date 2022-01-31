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
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

def difference_series(df):
    # Original Series
    fig, axes = plt.subplots(2, 2, sharex=True)
    axes[0, 0].plot(np.log(df['Price(USD)'])); axes[0, 0].set_title('Original Series')
    plot_acf(np.log(df['Price(USD)']), ax=axes[0, 1])
    
    # 1st Differencing
    axes[1, 0].plot(np.log(df['Price(USD)']).diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(np.log(df['Price(USD)']).dropna(), ax=axes[1, 1])
    

    return



def find_order_of_differencing(df):
    result = adfuller(df['Price(USD)'])
    adf_statistic = result[0]
    p_value = result[1]
    print('ADF Statistic: %f' % adf_statistic)
    print('p-value: %f' % p_value)
    
    if p_value > 0.05:
        difference_series(df)
    return
    

def eda(df):
    d = find_order_of_differencing(df)
    return
    



if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    
    # perform eda - define P,D,Q
    df = eda(df)
    