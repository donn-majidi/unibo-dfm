## In this script we define a function to transform series based on their transformation codes to achieve stationarity
## Reference: FRED-QD: A Quarterly Database for Macroeconomic Research by McCracken and Ng, 2020
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def fred_transform (x):
    transformed = []
    for i in range(x.shape[1]):
        series = x.iloc[2:,i]
        transform_code = x.iloc[1,i]
        if (transform_code == 2):
            series = series.diff()
        elif (transform_code == 3):
            series = series.diff().diff()
        elif (transform_code == 4):
            series = np.log(series)
        elif (transform_code == 5):
            series = np.log(series).diff()
        elif (transform_code == 6):
            series = np.log(series).diff().diff()
        elif (transform_code == 7):
            series = series/series.shift(1) - 1
        series.name = x.columns[i]
        #print(type(series))
        transformed.append(series)
    return pd.concat(transformed, axis = 1)
##once the series are transformed, we run the adf test to ensure that the series are indeed stationary
def adf_test(series,title=''):
    result = adfuller(series.dropna(),autolag='BIC')
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    return out
