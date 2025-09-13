import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime
#load data
data=pd.read_csv("Nat_Gas.csv")
#ensure date column is datetime
data['Dates']=pd.to_datetime(data['Dates'])
data.set_index('Dates',inplace=True)
#sort by date
data=data.sort_index()
#visualize historical data
plt.figure(figsize=(10,5))
plt.plot(data.index,data['Prices'],label='historical price')
plt.title("natural gas monthly prices")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.legend()
plt.show()
#------
#fit arima model for forecasting
#------
model=ARIMA(data['Prices'],order=(2,1,2))
model_fit=model.fit()
#------
#forecast 12 months into the future
forecast=model_fit.forecast(steps=12)
forecast_dates=pd.date_range(start=data.index[-1]+pd.offsets.monthEnd(),periods=12,freq='M')
forecast_series=pd.Series(forecast,index=forecast_dates)
#combine historical +forecast
all_data=pd.contact([data['Prices'],forecast_series])
#define function to estimate price 
# -------------------------
def estimate_price(date_str):
    """
    Estimate natural gas price for a given date.
    date_str: 'YYYY-MM-DD'
    """
    date = pd.to_datetime(date_str)

    if date in all_data.index:
        return all_data.loc[date]

    # If within range, interpolate
    if all_data.index[0] <= date <= all_data.index[-1]:
        return np.interp(
            date.value,
            all_data.index.view(np.int64),
            all_data.values
        )

    return None  # If outside forecast range


# -------------------------
# 5. Example Usage
# -------------------------
print("Price on 2022-06-15:", round(estimate_price("2022-06-15"), 2))
print("Price on 2025-05-30 (forecast):", round(estimate_price("2025-05-30"), 2))

# -------------------------
# 6. Plot with Forecast
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Price'], label='Historical')
plt.plot(forecast_series.index, forecast_series, label='Forecast', linestyle='--')
plt.title("Natural Gas Price Forecast")
plt.xlabel("Date")
plt.ylabel("Prices")
plt.legend()
plt.show() 