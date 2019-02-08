import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from fbprophet import Prophet

df = pd.read_csv('../input/Regression/AMZN.csv', parse_dates=['Date'])
print(df.head())
df = df.drop(['Open', 'High', 'Low', 'Volume', 'Name'], axis=1)
df.rename(columns={'Date': 'ds',
                   'Close': 'y'},
          inplace=True)

print(df.head())


df2 = df.asfreq(freq='M') # asfreq method is used to convert a time series to a specified frequency. Here it is monthly frequency.
print(df2.head())
# plt.title('Humidity in Kansas City over time(Monthly frequency)')
# plt.show()

# month_mean = df.groupby(df.ds.dt.).mean();

# print(month_mean)

m = Prophet()
m.fit(df)

# Make a future dataframe for 2 years
m_forecast = m.make_future_dataframe(periods=365, freq='D')

# Make predictions
m_forecast = m.predict(m_forecast)

print(m_forecast.head())
print(m_forecast.tail())

fig = m.plot(m_forecast, xlabel = 'Date', ylabel = 'Stock Price')
plt.title('Market Cap');
plt.show()

m.plot_components(m_forecast)
plt.show()