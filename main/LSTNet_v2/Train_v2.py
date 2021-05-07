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

# from main.LSTNet_v1.utils_v1 import related_trends
# from main.LSTNet_v1.Dataloader_v1 import BaseCOVDataset
# from main.LSTNet_v1.LSTNet_v1 import LSTNet_v1

from utils_v1 import related_trends
from Dataloader_v1 import BaseCOVDataset
from LSTNet_v1 import LSTNet_v1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import argparse
parser = argparse.ArgumentParser(description = 'Training model')
parser.add_argument('--GT_trends', default=None, type=str,
                    help='Define which Google Trends terms to use: all, related_average, or primary (default)')
parser.add_argument('--batch_size', default=3, type=int,
                    help='Speficy the bath size for the model to train to')
parser.add_argument('--epochs', default=100, type=int,
                    help='Speficy the epochs the model to train for')
parser.add_argument('--learning_rate', default=0.01, type=float,
                    help='Speficy the learning reate')
args = parser.parse_args()

#--------------------------------------------------------------------------------------------------
#----------------------------------------- Train functions ----------------------------------------

def train(model, dataloader, epochs, optimizer, criterion, savename, device):
    model.train()
    train_loss_list = []
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} of {epochs}')
        epoch_loss_train = 0
        for i, batch in tqdm(enumerate(dataloader, start=1), 
                            leave=False, desc="Train", total=len(dataloader)):
                
            X, Y = batch
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()

            with open(os.path.join(main_path,'main','LSTNet_v1','Log/Running-Loss.txt'), 'a+') as file:
                file.write(f'{loss.item()}\n')
            epoch_loss_train += loss.item()
            
        epoch_loss_train = epoch_loss_train / len(dataloader)
        train_loss_list.append(epoch_loss_train)
        
        with open(os.path.join(main_path,'main','LSTNet_v1','Log/Epoch-Loss.txt'), 'a+') as file:
            file.write(f'{epoch_loss_train}\n')

        print('Train loss: {:.3f}'.format(epoch_loss_train))

    # Save model
    model_name = f'LSTNet_v1_epochs_{epochs}_{savename}.pth'
    save_path  = os.path.join(main_path,'main','LSTNet_v1','Models',model_name)
    torch.save(model.state_dict(), save_path)

    return train_loss_list

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
if args.GT_trends == None:
    data_GT = pd.read_csv(data_GT_path, usecols=['date_time','anosmia','fiebre','covid','neumonia','sintomas covid'])
elif args.GT_trends == 'all':
    data_GT = pd.read_csv(data_GT_path)
    data_GT = data_GT.drop('Unnamed: 0', axis=1)
# elif args.GT_trends == 'related_average':
#     r_select = [0,2,3,4,5,7,10,12,13,16,17,19,24,25]
#     data_main_terms   = pd.read_csv(data_GT_id_terms_path)
#     data_search_terms = pd.read_csv(data_GT_search_terms_path)
#     data_GT = pd.read_csv(data_GT_path)
#     data_GT = related_trends(data_search_terms, data_main_terms, data_GT, r_select)
data_GT['date_time'] = pd.to_datetime(data_GT['date_time'], format='%Y-%m-%d')
data_GT = data_GT[data_GT['date_time'] <= last_cases_conf_date]
data_GT = data_GT.set_index('date_time')
data_GT = data_GT.rolling(window=7).mean()

### Concatenate all data
data_all = pd.concat([data_cases, data_movement_change, data_GT])
# data_all = pd.concat([data_cases, data_GT])
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
#--------------------------------------- Train dataset --------------------------------------------

data_input = data_all.drop(columns=['num_cases','num_diseased'], axis=1)
train_end_time = datetime.strptime('2021-04-04','%Y-%m-%d')  # train until last 14 days
train_data = data_input[data_input['date_time'] < train_end_time].copy()

## Min-Max Normalization
min_cases = train_data['num_cases_7dRA'].min() ; max_cases = train_data['num_cases_7dRA'].max()
min_diseased = train_data['num_diseased_7dRA'].min() ; max_diseased = train_data['num_diseased_7dRA'].max()
min_movement = train_data['movement_change_7dRA'].min() ; max_movement = train_data['movement_change_7dRA'].max()
min_anosmia = train_data['anosmia'].min() ; max_anosmia = train_data['anosmia'].max()
# min_tos = train_data['tos'].min() ; max_tos = train_data['tos'].max()
min_fiebre = train_data['fiebre'].min() ; max_fiebre = train_data['fiebre'].max()
min_covid = train_data['covid'].min() ; max_covid = train_data['covid'].max()

train_data.loc[:,'num_cases-N'] = (train_data['num_cases_7dRA']-min_cases)/(max_cases-min_cases)
train_data.loc[:,'num_diseased-N'] = (train_data['num_diseased_7dRA']-min_diseased)/(max_diseased-min_diseased)
train_data.loc[:,'movement-N'] = (train_data['movement_change_7dRA']-min_movement)/(max_movement-min_movement)
train_data.loc[:,'anosmia-N'] = (train_data['anosmia']-min_anosmia)/(max_anosmia-min_anosmia)
# # # train_data.loc[:,'tos-N'] = (train_data['tos']-min_tos)/(max_tos-min_tos)
train_data.loc[:,'fiebre-N'] = (train_data['fiebre']-min_fiebre)/(max_fiebre-min_fiebre)
train_data.loc[:,'covid-N'] = (train_data['covid']-min_covid)/(max_covid-min_covid)

## Create DataLoader
train_dataset = BaseCOVDataset(train_data, history_len=18)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

#--------------------------------------------------------------------------------------------------
#--------------------------------------- Training model -------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTNet_v1()
# model = model.to(device)

epochs = args.epochs
lr = args.learning_rate
weight_decay = 0.01

criterion = nn.MSELoss() ; savename = 'MSE'
# criterion = nn.L1Loss() ; savename = 'L1'
optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

print('\nTraining model...')
train_loss = train(model,train_data_loader, epochs, optimizer, criterion, savename, device)
joblib.dump(train_loss, os.path.join(main_path,'main','LSTNet_v1','Log',f'Training_Loss_epochs_{epochs}_{savename}.pkl'))

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='b', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(os.path.join(main_path,'main','LSTNet_v1','Log','Figures',
                                f'Training_Loss_epochs_{epochs}_{savename}.png'))
