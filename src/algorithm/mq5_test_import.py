import MetaTrader5 as mt5
from datetime import datetime

# Initialize MT5
if not mt5.initialize():
    print("MetaTrader 5 initialization failed")
    mt5.shutdown()
    exit()

# Define symbol and timeframe
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_H1
from_date = datetime(2020, 1, 1)
to_date = datetime(2020, 12, 31)

# Request historical data
rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

# Check and print the results
if rates is not None:
    for rate in rates:
        print(rate)
else:
    print("No data received")

# Shutdown MetaTrader 5
mt5.shutdown()