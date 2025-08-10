# Import libraries and FRED datareader
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
from datetime import datetime

# Define start and end dates
start = datetime(1982, 1, 1)
end = datetime(2022, 9, 30)

# NBER business cycle classification
recession = pdr.DataReader('USREC', 'fred', start, end)

# Percentage of time the US economy was in recession since 1982
recession_percentage = round(recession['USREC'].sum() / recession['USREC'].count() * 100, 2)

print(f"Percentage of time the US economy was in recession since 1982: {recession_percentage}%")
