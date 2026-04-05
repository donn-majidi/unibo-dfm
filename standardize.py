##once the series are transformed, and non-stationary series detected, we remove the non-stationary series from the data and standardize
import numpy as np
import pandas as pd
from src.python.data_loader import fred_transform
from src.python.data_loader import adf_test

df = pd.read_csv('./data/2024-10.csv')
X = fred_transform(df)
## we loose the first two rows because of differencing and second order differencing
X = X.iloc[2:,]
X.reset_index(drop=True, inplace=True)
X['sasdate'] = pd.to_datetime(X['sasdate'])
## finally, we retain observations only up to 2019Q4 before the outbreak of the COVID19 pandemic
X.drop(X.tail(19).index, inplace=True)

nonstat = []
for i in range(1,X.shape[1]):
    series = X.iloc[:,i]
    adf_result = adf_test(series)
    if (adf_result['p-value'] > 0.05):
        nonstat.append(series.name)

print(f'These are the non-stationary series\n\n{nonstat}\n\n')

X.drop(labels = nonstat, axis = 1, inplace = True)

print(X.head(5))

X.to_csv('./data/original_stationary_data.csv')

print(X.columns.get_loc('GDPC1')+1)
print(X.columns.get_loc('PCECC96')+1)
print(X.columns.get_loc('UNRATE')+1)
print(X.columns.get_loc('DPIC96')+1)
print(X.columns.get_loc('INDPRO')+1)
print(X.columns.get_loc('HOUST')+1)
print(X.columns.get_loc('TB3MS')+1)
