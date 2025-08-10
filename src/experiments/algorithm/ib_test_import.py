from ib_insync import *
# from datetime import datetime

# Connect to Interactive Brokers (IBKR TWS or IB Gateway)
ib = IB()
# ib.connect('127.0.0.1', 7497, clientId=1)
#
# # Define the options contract for Apple (AAPL)
# option = Option('AAPL', '20231215', 150, 'C', 'SMART')  # Example: Call option expiring on Dec 15, 2023, with a 150 strike price
#
# # Request historical data for the option
# data = ib.reqHistoricalData(
#     option,
#     endDateTime=datetime.now(),  # Current time
#     durationStr='1 M',  # Last 1 month of data
#     barSizeSetting='1 hour',  # 1-hour candlesticks
#     whatToShow='MIDPOINT',  # MIDPOINT price
#     useRTH=True,  # Use regular trading hours
#     formatDate=1
# )
#
# # Print the data retrieved
# for bar in data:
#     print(bar)
#
# # Disconnect from the IBKR API
# ib.disconnect()