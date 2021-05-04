import os
import sys
import joblib
# sys.path.append('../')
main_path = os.path.split(os.getcwd())[0] + '/covid19_forecast_ml'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm

from Dataloader_v1 import BaseCOVDataset
from LSTNet_v1 import LSTNet_v1

import torch
from torch.utils.data import Dataset, DataLoader

import argparse
parser = argparse.ArgumentParser(description = 'Training model')
parser.add_argument('--GT_trends', default=None, type=str,
                    help='Define which Google Trends terms to use: all, related_average, or primary (default)')
parser.add_argument('--batch_size', default=3, type=int,
                    help='Speficy the bath size for the model to train to')
parser.add_argument('--model_load', default='LSTNet_v1_epochs_100', type=str,
                    help='Define which model to evaluate')

args = parser.parse_args()


#--------------------------------------------------------------------------------------------------
#----------------------------------------- Test functions ----------------------------------------

def predict(model, dataloader, min_cases, max_cases):
    model.eval()
    predictions = None
    for i, batch in tqdm(enumerate(dataloader, start=1),leave=False, total=len(dataloader)):
                
        X, Y = batch
        Y_pred = model(X).detach().numpy()
        if i == 1:
            predictions = Y_pred
        else:
            predictions = np.concatenate((predictions, Y_pred), axis=0)
    predictions = predictions*(max_cases-min_cases)+min_cases
    columns = ['forecast_cases']
    df_predictions = pd.DataFrame(predictions, columns=columns)
    return df_predictions

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Data paths ---------------------------------------------

data_cases_path = os.path.join('data','cases.csv')
data_movement_change_path = os.path.join('data','movement_range.csv')
data_GT_path = os.path.join('data','Google_Trends','trends_BOG.csv')
data_GT_id_terms_path = os.path.join('data','Google_Trends','terms_id_ES.csv')
data_GT_search_terms_path = os.path.join('data','Google_Trends','search_terms_ES.csv')

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Load data ----------------------------------------------

### Load confirmed cases for Bogota
data_cases = pd.read_csv(data_cases_path, usecols=['date_time','location','num_cases','num_diseased'])
data_cases = data_cases.query(" location == 'Bogotá D.C.-Bogotá d C.' ")
data_cases = data_cases.drop('location', axis=1)
data_cases['date_time'] = pd.to_datetime(data_cases['date_time'], format='%Y-%m-%d')    # converted to datetime
data_cases = data_cases.set_index('date_time')
# Smooth data
data_cases['num_cases_7dRA']    = data_cases['num_cases'].rolling(window=7).mean()
data_cases['num_diseased_7dRA'] = data_cases['num_diseased'].rolling(window=7).mean()

### Load mobility data for Bogota
data_movement_change = pd.read_csv(data_movement_change_path, parse_dates=['date_time']).set_index('poly_id')
data_movement_change = data_movement_change.loc[11001].sort_values(by='date_time')
# Smooth data 
data_movement_change['movement_change_7dRA'] = data_movement_change['movement_change'].rolling(window=7).mean()
#data_movement_change['movement_change_7dRA'].iloc[:6] = data_movement_change['movement_change'].iloc[:6]
data_movement_change = data_movement_change.reset_index() ; data_movement_change = data_movement_change.set_index('date_time')
data_movement_change = data_movement_change.drop('poly_id', axis=1)

### Load Google Trends data for Bogota
if args.GT_trends == None:
    data_GT = pd.read_csv(data_GT_path, usecols=['date','anosmia','tos','fiebre','covid'])
elif args.GT_trends == 'all':
    data_GT = pd.read_csv(data_GT_path)
    data_GT = data_GT.drop('Unnamed: 0', axis=1)
# elif args.GT_trends == 'related_average':
#     r_select = [0,2,3,4,5,7,10,12,13,16,17,19,24,25]
#     data_main_terms   = pd.read_csv(data_GT_id_terms_path)
#     data_search_terms = pd.read_csv(data_GT_search_terms_path)
#     data_GT = pd.read_csv(data_GT_path)
#     data_GT = related_trends(data_search_terms, data_main_terms, data_GT, r_select)
data_GT = data_GT.rename(columns={'date':'date_time'})
data_GT['date_time'] = pd.to_datetime(data_GT['date_time'], format='%Y-%m-%d')
data_GT = data_GT.set_index('date_time')
data_GT = data_GT.rolling(window=7).mean()

### Concatenate all data
# data_all = pd.concat([data_cases, data_movement_change, data_GT])
data_all = pd.concat([data_cases, data_GT])
data_all = data_all.reset_index()
data_all = data_all.sort_values(by=['date_time'])
data_all = data_all.groupby('date_time').first().reset_index()
data_all = data_all.ffill()  # fill missing values (NaNs)
data_all = data_all.dropna().reset_index()  # eliminating NaNs
data_all = data_all.drop('index',axis=1)

start_dt = data_all.loc[0, 'date_time']
end_dt   = data_all.loc[len(data_all)-1, 'date_time']
print(f'Data from {start_dt} to {end_dt}')


#--------------------------------------------------------------------------------------------------
#----------------------------------------- Load model ----------------------------------------------

model_path = os.path.join(main_path,'main','LSTNet_v1','Models','{}.pth'.format(args.model_load))
model = LSTNet_v1()
model.load_state_dict(torch.load(model_path))

#--------------------------------------------------------------------------------------------------
#--------------------------------------- Test dataset --------------------------------------------

data_input = data_all.drop(columns=['num_cases','num_diseased'], axis=1)
## Min-Max Normalization
min_cases = data_input['num_cases_7dRA'].min() ; max_cases = data_input['num_cases_7dRA'].max()
min_diseased = data_input['num_diseased_7dRA'].min() ; max_diseased = data_input['num_diseased_7dRA'].max()
min_anosmia = data_input['anosmia'].min() ; max_anosmia = data_input['anosmia'].max()
min_tos = data_input['tos'].min() ; max_tos = data_input['tos'].max()
min_fiebre = data_input['fiebre'].min() ; max_fiebre = data_input['fiebre'].max()
min_covid = data_input['covid'].min() ; max_covid = data_input['covid'].max()

data_input.loc[:,'num_cases-N'] = (data_input['num_cases_7dRA']-min_cases)/(max_cases-min_cases)
data_input.loc[:,'num_diseased-N'] = (data_input['num_diseased_7dRA']-min_diseased)/(max_diseased-min_diseased)
data_input.loc[:,'anosmia-N'] = (data_input['anosmia']-min_anosmia)/(max_anosmia-min_anosmia)
data_input.loc[:,'tos-N'] = (data_input['tos']-min_tos)/(max_tos-min_tos)
data_input.loc[:,'fiebre-N'] = (data_input['fiebre']-min_fiebre)/(max_fiebre-min_fiebre)
data_input.loc[:,'covid-N'] = (data_input['covid']-min_covid)/(max_covid-min_covid)


dataset = BaseCOVDataset(data_input, history_len=18)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


#--------------------------------------------------------------------------------------------------
#----------------------------------------- Test model ---------------------------------------------

pred_forecast = predict(model, data_loader, min_cases, max_cases)

pred_forecast['date_time'] = pd.Series(pred_forecast.index).apply(lambda x: \
                                    data_all['date_time'][x+18])
pred_forecast = pred_forecast[['date_time','forecast_cases']]

#--------------------------------------------------------------------------------------------------
#-------------------------------------- Viz predictions -------------------------------------------

time_mask = datetime.strptime('2021-03-28','%Y-%m-%d')
data_mask = data_input['date_time'] >= time_mask
actual_data = data_input[data_mask].copy()
actual_data['date_time'] = pd.to_datetime(actual_data['date_time'], format='%Y-%m-%d')
pred_mask = pred_forecast['date_time'] >= time_mask
predict_data = pred_forecast[pred_mask].copy()
predict_data['date_time'] = pd.to_datetime(predict_data['date_time'], format='%Y-%m-%d')

# Calculate MSE
true_arr = np.array(actual_data['num_cases_7dRA'].tolist())
pred_arr = np.array(predict_data['forecast_cases'].tolist())
difference_array = np.subtract(true_arr, pred_arr)
squared_array = np.square(difference_array)
mse = squared_array.mean()

print('MSE: ' + str(mse))

# Plot
fig, ax = plt.subplots(1,1,figsize=(10,7))
ax.plot(actual_data['date_time'],actual_data['num_cases_7dRA'], color='k', label='Actual')
ax.plot(predict_data['date_time'],predict_data['forecast_cases'], color='r', label='Predicted')
ax.legend(loc='best', fontsize=15)
ax.set_xlabel('Days', fontsize=18)
ax.set_ylabel('Number of cases', fontsize=18)
plt.savefig(os.path.join(main_path,'main','LSTNet_v1','Log','Figures',
                                f'Predictions_{args.model_load}.png'))

print('end')