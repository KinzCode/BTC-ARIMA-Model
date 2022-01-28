# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:25:58 2022

@author: KinzCode
"""
import requests
import json
import datetime
import pandas as  pd


def get_new_prices(historical_data):
    max_date = pd.to_datetime(historical_data['Date']).max()
    today_date = pd.to_datetime("today")
    date_difference = (today_date - max_date).days
    print(max_date, today_date, date_difference)
    chosen_currency = historical_data['Currency'][0]
    
    #get_historical_prices(chosen_currency, date_difference, False)
    return


def get_historical_prices(chosen_currency, num_days, first_parse):
    """
    Parameters
    ----------
    chosen_currency : STR
        Provide a valid crypto currency e.g. 'bitcoin'.
    num_days : INT
        Enter the number of days history wanted.
    Returns
    -------
    data : DF
        Returns df of dates and the historical prices.

    """
    
    response = requests.get(f'https://api.coingecko.com/api/v3/coins/{chosen_currency}/market_chart?vs_currency=usd&days={num_days}&interval=daily')
    hist_dict = response.json()
    
    data = pd.DataFrame.from_dict(hist_dict['prices'])
    data.rename(columns = {0: 'Date', 1: 'Price(USD)'}, inplace = True)
    data['Date'] = pd.to_datetime(data['Date'], unit = 'ms')
    data['Date'] = data['Date'].dt.date
    data['Currency'] = chosen_currency
    
    if first_parse is False:
        print("and here")
        data.to_csv(f'../dat/{chosen_currency}_daily_historical.csv', mode='a', header=False, index = False)
    else:
        data.to_csv(f'../dat/{chosen_currency}_daily_historical.csv', index = False)


if __name__ == '__main__':
    # adjust desired currency here
    chosen_currency = 'bitcoin'
    try:
        historical_data = pd.read_csv(f'../dat/{chosen_currency}_daily_historical.csv')
    except FileNotFoundError:
        historical_data  = pd.DataFrame()
    
    if len(historical_data) > 0:
        get_new_prices(historical_data)
    else:
        get_historical_prices(chosen_currency, 3650, True)

