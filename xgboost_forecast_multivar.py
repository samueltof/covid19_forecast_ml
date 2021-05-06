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
scaler = StandardScaler()
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, RidgeCV
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
#----------------------------------------- Train model ----------------------------------------------


# Modeling and getting the metric results

data = pd.DataFrame(data_all.num_cases_7dRA)
data.columns = ["y"]

# Drop data after 2019-06
data = data.loc[data.index[:-1]]

# Adding the lag of the target variable from 1 steps back up to 24 days ago
for i in range(1, 24):
    data["lag_{}".format(i)] = data.y.shift(i)
    
y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

# Reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

# Scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    
# XGB
xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bynode=1, colsample_bytree=0.3, gamma=0,
        importance_type='gain', learning_rate=0.01, max_delta_step=0,
        max_depth=4, min_child_weight=1, missing=None, n_estimators=100,
        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
        silent=None, subsample=0.5, verbosity=1)   
xgb.fit(X_train_scaled, y_train)
prediction_xgb = xgb.predict(X_test_scaled)
error_xgb = mean_absolute_percentage_error(prediction_xgb, y_test)

# LightGBM
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_eval = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)
lightgbm_params = {'boosting_type': 'gbdt', 
        'colsample_bytree': 0.65, 
        'learning_rate': 0.001, 
        'n_estimators': 20, 
        'num_leaves': 3, 
        'reg_alpha': 0.5, 
        'reg_lambda': 0.5, 
        'subsample': 0.7}
gbm = lgb.train(lightgbm_params,lgb_train,num_boost_round=10,valid_sets=lgb_eval)
prediction_lgb = gbm.predict(X_test_scaled)
error_lightgbm = mean_absolute_percentage_error(prediction_lgb, y_test)

# Saving



#%% Plot

data = pd.DataFrame(data_all.num_cases_7dRA)
data.columns = ["y"]

# Drop data after 2019-06
data = data.loc[data.index[:-1]] 

# Adding the lag of the target variable from 7 steps back up to 48 months ago
for i in range(1, 14):
    data["lag_{}".format(i)] = data.y.shift(i)

y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

# Reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
# Scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGB
xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bynode=1, colsample_bytree=0.3, gamma=0,
        importance_type='gain', learning_rate=0.1, max_delta_step=0,
        max_depth=4, min_child_weight=1, missing=None, n_estimators=100,
        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
        silent=None, subsample=0.5, verbosity=1)          
xgb.fit(X_train_scaled, y_train)
prediction_xgb = xgb.predict(X_test_scaled)
error_xgb = mean_absolute_percentage_error(prediction_xgb, y_test)

# LightGBM
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_eval = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)
lightgbm_params = {'boosting_type': 'gbdt', 
            'colsample_bytree': 0.90, 
            'learning_rate': 0.005, 
            'n_estimators': 40, 
            'num_leaves': 6, 
            'reg_alpha': 1, 
            'reg_lambda': 1, 
            'subsample': 0.7}
gbm = lgb.train(lightgbm_params,lgb_train,num_boost_round=10,valid_sets=lgb_eval)
prediction_lgb = gbm.predict(X_test_scaled)
error_lightgbm = mean_absolute_percentage_error(prediction_lgb, y_test)

# Prediction
data_plot = pd.DataFrame(data_all.num_cases_7dRA).reset_index()
y_test_plot = y_test.reset_index()
y_hat_plot = pd.DataFrame()
y_hat_plot['index'] = y_test_plot['index']
y_hat_plot['predict_xgb'] = prediction_xgb
y_hat_plot['predict_lgb'] = prediction_lgb

# Plot
fig, ax = plt.subplots(1,1)
ax.plot(data_plot['index'],data_plot['num_cases_7dRA'], label='True')
ax.plot(y_hat_plot['index'],y_hat_plot['predict_xgb'], label='XGB')
ax.plot(y_hat_plot['index'],y_hat_plot['predict_lgb'], label='LGB')
ax.legend()
plt.show()



# %%

# Data preprocessing
data = pd.DataFrame(data_all.num_cases_7dRA)
data.columns = ["y"]
data = data.loc[data.index[:-1]]
X_predict = pd.DataFrame(index = ['2020-04-28', '2020-04-29', '2020-04-30', '2020-05-01', '2020-05-02', '2020-05-03'])
data = pd.concat([data,X_predict])
for i in range(1, 14):
    data["lag_{}".format(i)] = data.y.shift(i)
data = data.reset_index()
data['index'] = pd.to_datetime(data['index'])
data['index'] = data['index'].apply(lambda x: x.strftime("%Y-%m-%d"))
data_train = data[data['index'] < '2020-04-28'].set_index('index')
data_test = data[data['index'] >= '2020-04-28'].set_index('index')
y_train = data_train.dropna().y
X_train = data_train.dropna().drop(['y'], axis=1)    
X_test = data_test.drop(['y'], axis=1)   
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# XGB
xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bynode=1, colsample_bytree=0.3, gamma=0,
        importance_type='gain', learning_rate=0.1, max_delta_step=0,
        max_depth=4, min_child_weight=1, missing=None, n_estimators=100,
        n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
        silent=None, subsample=0.5, verbosity=1)          
xgb.fit(X_train_scaled, y_train)
prediction_xgb = xgb.predict(X_test_scaled)

# LightGBM
lgb_train = lgb.Dataset(X_train_scaled, y_train)
# lgb_eval = lgb.Dataset(X_test_scaled, y_test, reference=lgb_train)
lightgbm_params = {'boosting_type': 'gbdt', 
            'colsample_bytree': 0.90, 
            'learning_rate': 0.005, 
            'n_estimators': 40, 
            'num_leaves': 6, 
            'reg_alpha': 1, 
            'reg_lambda': 1, 
            'subsample': 0.7}
gbm = lgb.train(lightgbm_params,lgb_train,num_boost_round=10)
prediction_lgb = gbm.predict(X_test_scaled)

# Stacking
final_df = pd.DataFrame(index = [['2020-04-28', '2020-04-29', '2020-04-30', '2020-05-01', '2020-05-02', '2020-05-03']] )
final_df['forecast_xgb'] = prediction_xgb
final_df['forecast_lgb'] = prediction_lgb
final_df = final_df.reset_index()
final_df = final_df.rename(columns={'level_0':'index'})

fig, ax = plt.subplots(1,1)
# ax.plot(data_all['date_time'],data_all['num_cases_7dRA'], label='True')
ax.plot(final_df['index'],final_df['forecast_xgb'], label='XGB')
ax.plot(final_df['index'],final_df['forecast_lgb'], label='LGB')
ax.legend()
plt.show()


# %%
