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
import numpy as np
import datetime

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
    
    time_series = np.log(df)
    # fit
    model = sm.tsa.arima.ARIMA(time_series, order = p_d_q)
    fitted = model.fit()

    
    fc = fitted.get_forecast(7) 
    #Set confidence to 95% 
    fc = (fc.summary_frame(alpha=0.05))
    #Get mean forecast
    fc_mean = fc['mean']
    #Get lower confidence forecast
    fc_lower = fc['mean_ci_lower']
    #Get upper confidence forecast
    fc_upper = fc['mean_ci_upper'] 
    #Set figure size
    plt.figure(figsize=(12,8), dpi=100)
    #Plot last 50 price movements
    plt.plot(orig_df['Date'][-50:],orig_df['Price(USD)'][-50:], label='BTC Price')
    # create date axis for predictions
    future_7_days =  [str(datetime.datetime.today() + datetime.timedelta(days=x)) for x in range(7)]
    #Plot mean forecas
    plt.plot(future_7_days, np.exp(fc_mean), label='mean_forecast', linewidth = 1.5)
    #Create confidence interval
    plt.fill_between(future_7_days, np.exp(fc_lower),np.exp(fc_upper), color='b', alpha=.1, label = '95% Confidence')
    #Set title
    plt.title('Bitcoin 7 Day Forecast')
    #Set legend
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    #import data
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    # copy df
    orig_df = df.copy()
    df = df['Price(USD)']
    
    #import auto arima p,q,q
    with open('auto_p_d_q.json', 'r') as fp:
        auto_p_d_q = json.load(fp)
    
    # conver p,d,q to tuple
    auto_p_d_q = tuple(auto_p_d_q.values())
    
    # fit and predict auto model
    model(df, auto_p_d_q)
    

    
    



