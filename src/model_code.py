# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:24:28 2022

@author: KinzCode
"""

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
    

# def auto_arima(df):
#     df = df['Price(USD)']
#     train = df[:3165]
#     test = df[3165:]
#     model = pm.auto_arima(train, start_p=1, start_q=1,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=15, max_q=15, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=False,   # No Seasonality
#                       start_P=0, 
#                       D=0, 
#                       trace=True,
#                       error_action='ignore',  
#                       suppress_warnings=True, 
#                       stepwise=False)
    

#     n_periods = 24
#     fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    
#     test.reset_index(drop = True, inplace = True)
#     plt.figure(figsize=(12,5), dpi=100)
#     plt.plot(test, label='actual')
#     plt.plot(fc, label='mean_forecast', linewidth = 1.5)
#     plt.plot(confint[:,0], label = 'mean_ci_lower')
#     plt.plot(confint[:,1], label = 'mean_ci_upper')
#     plt.title('Forecast vs Actuals')
#     plt.legend(loc='upper left', fontsize=8)
#     plt.show()
    
#     return fc, confint


if __name__ == '__main__':
    # enter selected currency
    selected_currency = 'bitcoin'
    df = pd.read_csv(f'../dat/clean/cleaned_{selected_currency}_daily_historical.csv')
    orig_df = df.copy()
    d = 1
    p = 3
    q = 3
    # fit ARIMA
    #fit_arima(orig_df)
    #fc = train_test_split(orig_df)
    #fc, confint = auto_arima(orig_df)