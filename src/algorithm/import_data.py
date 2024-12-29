import datetime

import pytz
import pandas as pd
import MetaTrader5 as mt5
import numpy as np

from src.algorithm.primal_functions import Primal

frame_M15 = mt5.TIMEFRAME_M15  # 15-minute time
frameframe_M30 = mt5.TIMEFRAME_M30  # 30-minute time frame
frame_H1 = mt5.TIMEFRAME_H1  # Hourly time frame
frame_H4 = mt5.TIMEFRAME_H4  # 4-hour time frame
frame_D1 = mt5.TIMEFRAME_D1  # Daily time frame
frame_W1 = mt5.TIMEFRAME_W1  # Weekly time frame
frame_M1 = mt5.TIMEFRAME_MN1  # Monthly time frame

class Import:

    @staticmethod
    def get_quotes( time_frame,year = 2005, month = 1, day = 1, asset ="EURUSD"):

        if not mt5.initialize():
            print("initialize() failed, error code", mt5.last_error())
            quit()
        timezone = pytz.timezone("Europe/Paris")
        time_from = datetime.datetime(year, month, day, tzinfo= timezone)
        time_to=datetime.datetime.now(timezone) + datetime.timedelta(days=1)

        rates = mt5.copy_rates_range(asset,time_frame, time_from, time_to)
        rates_frame = pd.DataFrame(rates)
        return rates_frame

    @staticmethod
    def add_column(data,times):
        for i in range(1,times+1):
            new = np.zeros((len(data),1), dtype = float)
            data = np.append(data,new,axis = 1)

    @staticmethod
    def mass_import(asset, time_frame):
        data = []
        if time_frame == 'H1':
            data = Import.get_quotes(frame_H1, 2013,1,1,asset= assets[asset])
            data = data.iloc[:,1:5].values
            data = data.round(decimals=5)
        if time_frame == 'D1':
            data = Import.get_quotes(frame_D1,2000,1,1, asset = assets[asset])
            data = data.iloc[:,1:5].values
            data = data.round(decimals=5)
        return data

now = datetime.datetime.now()
assets = ['EURUSD','USDCHF','GBPUSD','ETHUSD','XAUUSD','USDCAD','BTCUSD','XAGUSD','SP500m','UK100']
my_data = Import.mass_import(5,'D1')
my_data2 = Import.mass_import(2, 'H1')

data_array = np.array(my_data)
