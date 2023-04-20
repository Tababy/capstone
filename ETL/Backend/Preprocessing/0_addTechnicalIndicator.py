import pandas as pd
import os
import numpy as np


def RSI(df, n=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return delta, avg_gain, avg_loss , rsi
def SMA(df, column, period):
    return df[column].rolling(period).mean()
def EMA(df, column, period):
    try:
        return df[column].ewm(span=period, adjust=False).mean()
    except KeyError:
        return df.ewm(span=period, adjust=False).mean()
def WMA(df, column, period):
    weights = np.arange(1, period+1)
    wma = df[column].rolling(period).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    return wma
def Bollinger_Bands(df, column, period, std):
    sma = SMA(df, column, period)
    std_dev = df[column].rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return sma, upper_band, lower_band
def MACD(df, column, fast_period, slow_period, signal_period):
    ema_fast = EMA(df, column, fast_period)
    ema_slow = EMA(df, column, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, column, signal_period)
    return macd_line, signal_line

def Stochastic_Oscillator(df, high_col, low_col, close_col, n=14):
    highest_high = df[high_col].rolling(n).max()
    lowest_low = df[low_col].rolling(n).min()
    k = ((df[close_col] - lowest_low) / (highest_high - lowest_low)) * 100
    return k

def ATR(df, high_col, low_col, close_col, n=14):
    tr1 = abs(df[high_col] - df[low_col])
    tr2 = abs(df[high_col] - df[close_col].shift())
    tr3 = abs(df[low_col] - df[close_col].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(n).mean()
    return atr

def ROC(df, column, n=12):
    roc = 100 * (df[column] / df[column].shift(n) - 1)
    return roc


def OBV(df, close_col, volume_col):
    change_in_volume = np.where(df[close_col] > df[close_col].shift(1), df[volume_col], np.where(df[close_col] < df[close_col].shift(1), -df[volume_col], 0))
    obv = change_in_volume.cumsum()
    return obv

def RVI(df, close_col, high_col, low_col, n=14):
    diff = df[close_col].diff(1)
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]
    up_chg_avg = up_chg.rolling(n).mean()
    down_chg_avg = down_chg.rolling(n).mean().abs()
    rvi = 100 * up_chg_avg / (up_chg_avg + down_chg_avg)
    return rvi

def CCI(df, high_col, low_col, close_col, n=20):
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    moving_average = typical_price.rolling(n).mean()
    mean_deviation = typical_price.rolling(n).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    cci = (typical_price - moving_average) / (0.015 * mean_deviation)
    return cci

def MFI(df, high_col, low_col, close_col, volume_col, n=14):
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    raw_money_flow = typical_price * df[volume_col]
    flow_direction = np.where(typical_price > typical_price.shift(1), 1, -1)
    money_flow_ratio = raw_money_flow.rolling(n).sum() / (df[volume_col].rolling(n).sum() * flow_direction)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

def PSAR(df, high_col, low_col, close_col, af=0.02, af_max=0.2):
    psar = df[close_col].copy()
    psar_bull = df[low_col].copy()
    psar_bear = df[high_col].copy()
    psar_dir = 1
    af_current = af
    for i in range(2, len(df)):
        if psar_dir == 1:
            psar[i] = psar[i-1] + af_current * (psar_bull[i-1] - psar[i-1])
        else:
            psar[i] = psar[i-1] + af_current * (psar_bear[i-1] - psar[i-1])
        if psar_dir == 1 and df[high_col][i] > psar[i]:
            psar_dir = -1
            psar[i] = psar_bear[i-1]
            psar_bull[i] = psar[i]
            af_current = af
        elif psar_dir == -1 and df[low_col][i] < psar[i]:
            psar_dir = 1
            psar[i] = psar_bull[i-1]
            psar_bear[i] = psar[i]
            af_current = af
        else:
            if psar_dir == 1:
                if df[low_col][i] < psar_bull[i-1]:
                    psar_bull[i] = df[low_col][i]
                    af_current = min(af_current + af, af_max)
                else:
                    psar_bull[i] = psar_bull[i-1]
            else:
                if df[high_col][i] > psar_bear[i-1]:
                    psar_bear[i] = df[high_col][i]
                    af_current = min(af_current + af, af_max)
                else:
                    psar_bear[i] = psar_bear[i-1]
    return psar

def Ichimoku(df, high_col, low_col, close_col ,suffix='_ichimoku'):
    nine_period_high = df[high_col].rolling(window=9).max()
    nine_period_low = df[low_col].rolling(window=9).min()
    tenkan_sen = (nine_period_high + nine_period_low) / 2

    period26_high = df[high_col].rolling(window=26).max()
    period26_low = df[low_col].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    period52_high = df[high_col].rolling(window=52).max()
    period52_low = df[low_col].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    chikou_span = df[close_col].shift(-26)

    ichimoku = pd.concat([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span], axis=1, keys=[f'Tenkan_Sen_{suffix}', f'Kijun_Sen_{suffix}', f'Senkou_Span_A_{suffix}', f'Senkou_Span_B_{suffix}', f'Chikou_Span_{suffix}'])
    return ichimoku

def ForceIndex(df, close_col, volume_col, n=13):
    force_index = df[close_col].diff(1) * df[volume_col]
    force_index = force_index.rolling(n).sum()
    return force_index

def UltimateOscillator(df, high_col, low_col, close_col, short_period=7, medium_period=14, long_period=28, weight1=4.0, weight2=2.0, weight3=1.0):
    min_low_or_prev_close = pd.concat([df[low_col], df[close_col].shift(1)], axis=1).min(axis=1)
    max_high_or_prev_close = pd.concat([df[high_col], df[close_col].shift(1)], axis=1).max(axis=1)
    buying_pressure = df[close_col] - min_low_or_prev_close
    true_range = max_high_or_prev_close - min_low_or_prev_close
    avg1 = buying_pressure.rolling(window=short_period).sum() / true_range.rolling(window=short_period).sum()
    avg2 = buying_pressure.rolling(window=medium_period).sum() / true_range.rolling(window=medium_period).sum()
    avg3 = buying_pressure.rolling(window=long_period).sum() / true_range.rolling(window=long_period).sum()
    ultimate_oscillator = 100 * ((weight1 * avg1) + (weight2 * avg2) + (weight3 * avg3)) / (weight1 + weight2 + weight3)
    return ultimate_oscillator

def KeltnerChannels(df, high_col, low_col, close_col, n=20, atr_multiplier=2):
    mid = df[close_col].rolling(n).mean()
    atr = ATR(df, high_col, low_col, close_col, n)
    upper = mid + (atr_multiplier * atr)
    lower = mid - (atr_multiplier * atr)
    return upper, mid, lower
def DonchianChannels(df, high_col, low_col, close_col, n=20):
    upper = df[high_col].rolling(window=n).max()
    lower = df[low_col].rolling(window=n).min()
    middle = (upper + lower) / 2
    return upper, middle, lower
def CMF(df, high_col, low_col, close_col, volume_col, n=20):
    mfm = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
    mfv = mfm * df[volume_col]
    cmf = mfv.rolling(n).sum() / df[volume_col].rolling(n).sum()
    return cmf
def PSARmod(df, high_col, low_col, close_col, af_start=0.02, af_increment=0.02, af_max=0.2):
    psar = df[close_col].copy()
    psar_direction = pd.Series([1] * len(df), index=df.index)
    psar_extreme_point = df[low_col][0]
    psar_af = af_start
    for i in range(1, len(df)):
        prev_psar = psar[i - 1]
        curr_close = df[close_col][i]
        curr_high = df[high_col][i]
        curr_low = df[low_col][i]
        if psar_direction[i - 1] == 1:
            if curr_low < psar_extreme_point:
                psar_direction[i] = -1
                psar[i] = psar_extreme_point
                psar_af = af_start
                psar_extreme_point = curr_high
            else:
                psar_direction[i] = 1
                psar[i] = prev_psar + psar_af * (psar_extreme_point - prev_psar)
                if curr_high > psar_extreme_point:
                    psar_extreme_point = curr_high
                    psar_af = min(psar_af + af_increment, af_max)
        else:
            if curr_high > psar_extreme_point:
                psar_direction[i] = 1
                psar[i] = psar_extreme_point
                psar_af = af_start
                psar_extreme_point = curr_low
            else:
                psar_direction[i] = -1
                psar[i] = prev_psar + psar_af * (psar_extreme_point - prev_psar)
                if curr_low < psar_extreme_point:
                    psar_extreme_point = curr_low
                    psar_af = min(psar_af + af_increment, af_max)
    return psar


def FibonacciRetracements(df, high_col, low_col, trend='uptrend', levels=[0, 23.6, 38.2, 50, 61.8, 100]):
    if trend == 'uptrend':
        start = df[low_col].idxmin()
        end = df[high_col].idxmax()
        trend_data = df.loc[start:end, :]
    elif trend == 'downtrend':
        start = df[high_col].idxmax()
        end = df[low_col].idxmin()
        trend_data = df.loc[start:end, :]
    else:
        raise ValueError("Invalid trend parameter: must be 'uptrend' or 'downtrend'")
    
    trend_high = trend_data[high_col].max()
    trend_low = trend_data[low_col].min()
    price_range = trend_high - trend_low
    
    retracement_series = pd.Series(np.nan, index=df.index, name='Fibonacci_Retracements')
    
    for level in levels:
        retracement_level = trend_high - (price_range * level / 100)
        mask = (df.index >= start) & (df.index <= end)
        retracement_series[mask] = retracement_level
    
    return retracement_series


def PivotPoints(df, high_col, low_col, close_col):
    pivot = (df[high_col] + df[low_col] + df[close_col]) / 3
    s1 = (pivot * 2) - df[high_col]
    s2 = pivot - (df[high_col] - df[low_col])
    r1 = (pivot * 2) - df[low_col]
    r2 = pivot + (df[high_col] - df[low_col])
    return pivot, s1, s2, r1, r2


def VWAP(df, high_col, low_col, close_col, volume_col):
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    vwap = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()
    return vwap

def ADL(df, high_col, low_col, close_col, volume_col):
    money_flow_multiplier = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
    money_flow_volume = money_flow_multiplier * df[volume_col]
    adl = money_flow_volume.cumsum()
    return adl

def calculate_technical_indicators(df):
    # Moving Averages
    df['SMA_10'] = SMA(df, 'close', 10)
    df['EMA_20'] = EMA(df, 'close', 20)
    df['WMA_10'] = WMA(df, 'close', 10)
    df['SMA_20'], df['Upper_BB'], df['Lower_BB'] = Bollinger_Bands(df, 'close', 20, 2)
    df['MACD'], df['Signal'] = MACD(df, 'close', 12, 26, 9)

    # Momentum Indicators
    df['Stochastic_Oscillator'] = Stochastic_Oscillator(df, 'high', 'low', 'close', 14)
    df['ROC_12'] = ROC(df, 'close', 12)
    df['ForceIndex_13'] = ForceIndex(df, 'close', 'volume', 13)
    df['Ultimate_Oscillator'] = UltimateOscillator(df, 'high', 'low', 'close')
    df['RVI_14'] = RVI(df, 'close', 'high', 'low', 14)
    df['delta'], df['avg_gain'], df['avg_loss'] ,  df['RSI_14']= RSI(df , n = 14)
    
    # Trend Indicators
    df['PSAR'] = PSAR(df, 'high', 'low', 'close')
    #df['Ichimoku'] = Ichimoku(df, 'high', 'low','close')
    
    ichimoku = Ichimoku(df, 'high', 'low', 'close')
    df = df.join(ichimoku)  # Join the Ichimoku DataFrame with the original DataFrame
    
    df['Donchian_High'], df['Donchian_Middle'], df['Donchian_Low'] = DonchianChannels(df, 'high', 'low', 'close')
    df['Fibonacci_Retracements'] = FibonacciRetracements(df, 'high', 'low')
    df['Pivot_Points'], df['Pivot_Support_1'], df['Pivot_Support_2'], df['Pivot_Resistance_1'], df['Pivot_Resistance_2'] = PivotPoints(df, 'high', 'low', 'close')
    df['Keltner_Upper'], df['Keltner_Middle'], df['Keltner_Lower'] = KeltnerChannels(df, 'high', 'low', 'close', 20)

    # Volume Indicators
    df['ADL'] = ADL(df, 'high', 'low', 'close', 'volume')
    df['MFI_14'] = MFI(df, 'high', 'low', 'close', 'volume', 14)
    df['VWAP'] = VWAP(df, 'high', 'low', 'close', 'volume')
    df['CMF'] = CMF(df, 'high', 'low', 'close', 'volume', 20)

    # Volatility Indicators
    df['ATR'] = ATR(df, 'high', 'low', 'close', 14)
    df['CCI_20'] = CCI(df, 'high', 'low', 'close', 20)
    df['OBV'] = OBV(df, 'close', 'volume')

    return df


def get_num_days_since_update():
    from datetime import datetime
    df = pd.read_csv('./../../Database/Futures_um/klines/Full_Data_2klines.csv')
    df.sort_values('open_time',inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True,inplace=True)
    last_entry_dt = (df['open_time'].iloc[-1])

    last_entry_datetime = datetime.strptime(last_entry_dt, '%Y-%m-%d %H:%M:%S')

    import datetime
    base = datetime.datetime.today()
    num_days_since_last_entry = str(base -  last_entry_datetime).split(' ')[0]
    return num_days_since_last_entry

def get_date_list():

    num_days_since_last_entry = get_num_days_since_update()
    
    import datetime
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(int(num_days_since_last_entry)+1)]
    return date_list


def quick_date_sanity_check():
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.sort_values('open_time',inplace=True)

# Calclulate Technical Indicators for each period of time
def calculate_technical_per_period():
    my_dict = {'df_min':pd.DataFrame,'df_5min':pd.DataFrame(),'df_10min':pd.DataFrame(),'df_15min':pd.DataFrame(),'df_30min':pd.DataFrame(),'df_hour':pd.DataFrame(),'df_2hour':pd.DataFrame(),'df_4hour':pd.DataFrame(),'df_8hour':pd.DataFrame(),'df_12hour':pd.DataFrame(),'df_day':pd.DataFrame(),'df_week':pd.DataFrame()}
    li = [df_min,df_5min,df_10min,df_15min,df_30min,df_hour,df_2hour,df_4hour,df_8hour,df_12hour,df_day,df_week]
    for i in range(len(li)):
        my_dict[list(my_dict.keys())[i]] =  calculate_technical_indicators(li[i])
    return my_dict

# Rename columns
def rename_cols():
    for i in ['df_min' , 'df_5min', 'df_10min' , 'df_15min' , 'df_30min' , 'df_hour' , 'df_2hour' , 'df_4hour', 'df_8hour','df_12hour','df_day','df_week']:
        period = i.split('_')[-1]
        rename_dict1 = {'Chikou_Span__ichimoku': f'Chikou_Span__ichimoku_{period}',
        'Senkou_Span_A__ichimoku': f'Senkou_Span_A__ichimoku_{period}',
        'Fibonacci_Retracements': f'Fibonacci_Retracements_{period}',
        'Donchian_Low': f'Donchian_Low_{period}',
        'Pivot_Resistance_2': f'Pivot_Resistance_2_{period}',
        'quote_volume': f'quote_volume_{period}',
        'OBV': f'OBV_{period}',
        'ADL': f'ADL_{period}',
        'Keltner_Lower': f'Keltner_Lower_{period}',
        'VWAP': f'VWAP_{period}',
        'Upper_BB': f'Upper_BB_{period}',
        'Pivot_Support_2': f'Pivot_Support_2_{period}',
        'ATR': f'ATR_{period}',
        'WMA_10': f'WMA_10_{period}',
        'Stochastic_Oscillator': f'Stochastic_Oscillator_{period}',
        'SMA_10': f'SMA_10_{period}',
        'Pivot_Points': f'Pivot_Points_{period}',
        'Donchian_Middle': f'Donchian_Middle_{period}',
        'CCI_20': f'CCI_20_{period}',
        'SMA_20': f'SMA_20_{period}',
        'Pivot_Resistance_1': f'Pivot_Resistance_1_{period}',
        'MACD': f'MACD_{period}',
        'Kijun_Sen__ichimoku': f'Kijun_Sen__ichimoku_{period}',
        'PSAR': f'PSAR_{period}',
        'Keltner_Middle': f'Keltner_Middle_{period}',
        'Donchian_High': f'Donchian_High_{period}',
        'Senkou_Span_B__ichimoku': f'Senkou_Span_B__ichimoku_{period}',
        'Lower_BB': f'Lower_BB_{period}',
        'Tenkan_Sen__ichimoku': f'Tenkan_Sen__ichimoku_{period}',
        'ROC_12': f'ROC_12_{period}',
        'Ultimate_Oscillator': f'Ultimate_Oscillator_{period}',
        'CMF': f'CMF_{period}',
        'Pivot_Support_1': f'Pivot_Support_1_{period}',
        'ForceIndex_13': f'ForceIndex_13_{period}',
        'MFI_14': f'MFI_14_{period}',
        'Signal': f'Signal_{period}',
        'RVI_14': f'RVI_14_{period}',
        'Keltner_Upper': f'Keltner_Upper_{period}',
        'index': f'index_{period}',
        'EMA_20': f'EMA_20_{period}'}
        (my_dict[i]).rename(columns=rename_dict1 , inplace=True)
        (my_dict[i]).rename(columns={'open':f'open_{period}'	,'high': f'high_{period}', 	'low':f'low_{period}'	, 'close':f'close_{period}',	'volume':f'volume_{period}'	,'close_time':f'close_time_{period}','quota_volume':f'quote_volume_{period}'	,'count':f'count_{period}',	'taker_buy_volume':f'taker_buy_volume_{period}'	,'taker_buy_quote_volume':f'taker_buy_quote_volume_{period}','ignore':f'ignore_{period}','RSI_14':f'RSI_14_{period}','RS':f'RS_{period}','upPrices':f'upPrices_{period}','down_Prices':f'downPrices_{period}','delta':f'delta_{period}','avg_gain':f'avg_gain_{period}','avg_loss':f'avg_loss_{period}'},inplace=True)
    return my_dict


def save_csv_for_each_period(init=False):
 

    for i in ['df_min' , 'df_5min', 'df_10min' , 'df_15min' , 'df_30min' , 'df_hour' , 'df_2hour' , 'df_4hour', 'df_8hour','df_12hour','df_day','df_week']:
        
        '''try:
            my_dict[i].drop(columns=['index'],inplace=True)
            my_dict[i].drop(columns=[f'index_{i[3:]}'],inplace=True)

        except Exception as e:
            print(e , i)'''
            
        if type(my_dict[i].index.tolist()[0]) == int:
            my_dict[i].reset_index(inplace=True,drop=True)

        else:
            my_dict[i].reset_index(inplace=True)

        # Append or save as a new csv
        if init:
            my_dict[i].to_csv(f'./../../Storage/TechnicalIndicators/{i}/{i[3:]}_data.csv',index=False)

        else:
            pruv = pd.read_csv(f'./../../Storage/TechnicalIndicators/{i}/{i[3:]}_data.csv')

            if my_dict[i]['open_time'].iloc[-1] > pd.to_datetime(pruv['open_time'].iloc[-1]):
                import datetime
                dt = date_list[-2]
                dt_midnight = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
                df_pruv = my_dict[i].loc[my_dict[i]['open_time']>=dt_midnight]
                df_pruv.to_csv(f'./../../Storage/TechnicalIndicators/{i}/{i[3:]}_data.csv',mode='a',index=False,header=False)
            else:
                print(f"Nothing was apendd for {i}")

date_list = get_date_list()
df = pd.read_csv('./../../Database/Futures_um/klines/Full_Data_2klines.csv')
quick_date_sanity_check()

df_min = df.resample('min' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_5min = df.resample('5min' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_10min = df.resample('10min' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_15min = df.resample('15min' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_30min = df.resample('30min' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_hour = df.resample('h' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_2hour = df.resample('2h' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_4hour = df.resample('4h' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_8hour = df.resample('8h' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_12hour = df.resample('12h' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_day = df.resample('d' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})
df_week = df.resample('w' ,on="open_time").agg({'open':'mean','high':'mean','low':'mean','close':'mean', 'volume':'sum','close_time':'mean','quote_volume':'sum','count':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum' ,'ignore':'sum'})#,'RSI':'mean','RS':'mean'})

my_dict = calculate_technical_per_period()

my_dict = rename_cols()

save_csv_for_each_period(init=False)

