#%%

# Import everything we need
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
from datetime import datetime as dt

from sklearn.model_selection import cross_val_score, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
scaler = StandardScaler()
import xgboost
from xgboost import XGBRegressor 
import lightgbm as lgb

from plotly import __version__ 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go
import chart_studio.plotly
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
init_notebook_mode(connected = True)

import warnings
warnings.filterwarnings('ignore')

#%%

# Split data in a different way
def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test

# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

# Metric
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% 

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Data paths ---------------------------------------------

data_cases_path = os.path.join('data','cases_localidades.csv')
data_movement_change_path = os.path.join('data','Movement','movement_range_colombian_cities.csv')
data_GT_path = os.path.join('data','Google_Trends','trends_BOG.csv')
data_GT_id_terms_path = os.path.join('data','Google_Trends','terms_id_ES.csv')
data_GT_search_terms_path = os.path.join('data','Google_Trends','search_terms_ES.csv')

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Load data ----------------------------------------------

# Until these date
date_th = datetime.strptime('2021-05-01','%Y-%m-%d')

### Load confirmed cases for Bogota
data_cases = pd.read_csv(data_cases_path, usecols=['date_time','location','num_cases','num_diseased'])
data_cases['date_time'] = pd.to_datetime(data_cases['date_time'], format='%Y-%m-%d')    # converted to datetime
# data_cases = data_cases[data_cases['date_time'] <= date_th]
last_cases_conf_date = data_cases['date_time'].iloc[-1]
data_cases = data_cases.groupby('date_time').sum()
# Smooth data
data_cases['num_cases_7dRA']    = data_cases['num_cases'].rolling(window=7).mean()
data_cases['num_diseased_7dRA'] = data_cases['num_diseased'].rolling(window=7).mean()

### Load mobility data for Bogota
data_movement_change = pd.read_csv(data_movement_change_path, parse_dates=['date_time']).set_index('poly_id')
data_movement_change = data_movement_change.loc[11001].sort_values(by='date_time')
# Smooth data 
data_movement_change['movement_change_7dRA'] = data_movement_change['movement_change'].rolling(window=7).mean()
#data_movement_change['movement_change_7dRA'].iloc[:6] = data_movement_change['movement_change'].iloc[:6]
data_movement_change = data_movement_change[data_movement_change['date_time'] <= last_cases_conf_date]
data_movement_change = data_movement_change.reset_index() ; data_movement_change = data_movement_change.set_index('date_time')
data_movement_change = data_movement_change.drop('poly_id', axis=1)

### Load Google Trends data for Bogota
data_GT = pd.read_csv(data_GT_path, usecols=['date_time','anosmia','fiebre','covid'])
data_GT['date_time'] = pd.to_datetime(data_GT['date_time'], format='%Y-%m-%d')
data_GT = data_GT[data_GT['date_time'] <= last_cases_conf_date]
data_GT = data_GT.set_index('date_time')
data_GT = data_GT.rolling(window=7).mean()

### Concatenate all data
data_all = pd.concat([data_cases, data_movement_change, data_GT]) # pd.concat([data_cases, data_GT])
data_all = data_all.reset_index()
data_all = data_all.sort_values(by=['date_time'])
data_all = data_all.groupby('date_time').first().reset_index()
data_all = data_all.ffill()  # fill missing values (NaNs)
data_all = data_all.dropna().reset_index()  # eliminating NaNs
data_all = data_all.drop('index',axis=1)

start_dt = data_all.loc[0, 'date_time']
end_dt   = data_all.loc[len(data_all)-1, 'date_time']
print(f'Data from {start_dt} to {end_dt}')


# %%

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Data model ----------------------------------------------

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
 
# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(np.asarray([testX]))
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# %%

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Train model ----------------------------------------------

series = data_all
values = data_all.num_cases_7dRA.values.tolist()
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=6, n_out=6)
# evaluate
# mae, y, yhat = walk_forward_validation(data, 24)
# print('MAE: %.3f' % mae)
# # plot expected vs preducted
# fig, ax = plt.subplots(1,1,figsize=(8,6))
# ax.plot(y, label='Expected')
# ax.plot(yhat, label='Predicted')
# ax.legend()
# plt.show()


#%%
#--------------------------------------------------------------------------------------------------
#----------------------------------------- Test model ----------------------------------------------

# transform the time series data into supervised learning
train = series_to_supervised(values, n_in=6, n_out=6)
# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]
# fit model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(trainX, trainy)
# construct an input for a new preduction
row = values[-6:]
# make a one-step prediction
yhat = model.predict(np.asarray([row]))
# %%
