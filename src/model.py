# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:24:28 2022

@author: KinzCode
"""

import pandas as pd
import json
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt



def model(df, p_d_q):
    """
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    p_d_q : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    df_length = len(df)
    train_length = df_length - 31

    train = df[:train_length]
    test = df[train_length:]
    
    # fit
    model = sm.tsa.arima.ARIMA(train, order = p_d_q)
    fitted = model.fit()

    
    fc = fitted.get_forecast(32) 
    fc = (fc.summary_frame(alpha=0.05))
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


if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    #import data
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    df = df['Price(USD)']
    #import auto arima p,q,q
    with open('auto_p_d_q.json', 'r') as fp:
        auto_p_d_q = json.load(fp)
    
    # conver p,d,q to tuple
    auto_p_d_q = tuple(auto_p_d_q.values())
    
    # fit and predict auto model
    model(df, auto_p_d_q)
    

    
    



