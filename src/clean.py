# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:56:01 2022

@author: KinzCode
"""

import pandas as pd


def remove_duplicate_data(df):
    """
    Parameters
    ----------
    df : DataFrame
        A dateframe where each row contains a unique data and price for the
        respective currency.

    Returns
    -------
    df : DataFrame
        Dataframe with no duplicate rows or 2 separate prices for 1 day.

    """
    df.drop_duplicates('Date', keep = 'last', inplace = True)
    df.drop_duplicates(inplace = True)
    return df

def clean(df):
    """
    Parameters
    ----------
    df : DataFrame
        A dateframe where each row contains a unique data and price for the
        respective currency.

    Returns
    -------
    df : DataFrame
        Cleaned dataframe with all preprocessing/ data wrangling applied.

    """
    df = remove_duplicate_data(df)
    return df

if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/raw/{selected_currency}_daily_historical.csv')
    
    # clean data
    df = clean(df)
    
    # save to csv
    df.to_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv', index = False)