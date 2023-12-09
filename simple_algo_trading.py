import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yfinance as YahooFinance
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS

data = YahooFinance.download(['COPEC.SN','CMPC.SN'], datetime(2015,1,1), datetime(2023,12,8), progress=False)['Adj Close']
data.columns=['Copec', 'CMPC']

print(data.head(10))

fig = px.line(data, title = 'price')
fig.show()

returns = data.pct_change().dropna()
print(returns)

# copec=YahooFinance.Ticker('COPEC.SN')
# copec_historical = copec.history(start=datetime(2015,1,1),end=datetime(2023,12,8))

# cmpc=YahooFinance.Ticker('CMPC.SN')
# cmpc_historical = cmpc.history(start=datetime(2015,1,1),end=datetime(2023,12,8))

# data=pd.merge(copec_historical['Close'],cmpc_historical['Close'],how='inner',on='Date',suffixes=('_copec','_cmpc'))
# print(data)

# in_sample=data[data.index<datetime(2022,1,1,tzinfo=tz.gettz('America/Santiago'))]
# out_sample=data[data.index>=datetime(2022,1,1,tzinfo=tz.gettz('America/Santiago'))]

# x_train, x_test, y_train, y_test = train_test_split(in_sample['Close_copec'],in_sample['Close_cmpc'],test_size=0.3,random_state=410)
# print(x_train)
# print(y_train)

# model=LinearRegression()
# model.fit(x_train,y_train)