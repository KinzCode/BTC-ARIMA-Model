# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:25:58 2022

@author: KinzCode
"""
import requests
import json
import datetime
import pandas as  pd



def get_historical_prices(currency, num_days):
    """
    Parameters
    ----------
    currency : STR
        Provide a valid crypto currency e.g. 'bitcoin'.
    num_days : INT
        Enter the number of days history wanted.
    Returns
    -------
    data : DF
        Returns df of dates and the historical prices.

    """
    
    response = requests.get(f'https://api.coingecko.com/api/v3/coins/{currency}/market_chart?vs_currency=usd&days={num_days}&interval=daily')
    hist_dict = response.json()
    
    data = pd.DataFrame.from_dict(hist_dict['prices'])
    data.rename(columns = {0: 'Date', 1: 'Price(USD)'}, inplace = True)
    data['Date'] = pd.to_datetime(data['Date'], unit = 'ms')
    data['Currency'] = currency
    return data


historical_data = get_historical_prices('bitcoin', 3650)