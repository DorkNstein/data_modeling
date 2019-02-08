import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from fbprophet import Prophet

df = pd.read_csv('../input/Regression/avocado.csv', parse_dates=['Date'])
print(df.head(3))
df = df.drop(['id', 'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region'], axis=1)
df.rename(columns={'Date': 'ds',
                   'AveragePrice': 'y'},
          inplace=True)

print(df.head(3))

# month_mean = df.groupby(df.ds.dt.).mean();

# print(month_mean)

m = Prophet()
m.fit(df)

# Make a future dataframe for 2 years
m_forecast = m.make_future_dataframe(periods=365, freq='D')

# Make predictions
m_forecast = m.predict(m_forecast)

print(m_forecast.head(3))
print(m_forecast.tail(3))

fig = m.plot(m_forecast, xlabel = 'Date', ylabel = 'Average Price')
plt.title('Market Cap of GM');
plt.show()

m.plot_components(m_forecast)
plt.show()