import yfinance as yf
import datetime
import pandas as pd
import pytz

BTC = yf.Ticker("BTC-USD")

history = BTC.history(interval="1m",period='7d')


dt = datetime.datetime.today()
dt_midnight = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
history_today = history.loc[(pd.to_datetime(history.index) >= dt_midnight.replace(tzinfo=pytz.UTC))]
history_today.to_csv('./../../Storage/VoiceAlerts/tables/history_today.csv',index=True)
history.to_csv('./../../Storage/VoiceAlerts/tables/history.csv',index=True)
print('yFinance Data Downloaded for the week!')

