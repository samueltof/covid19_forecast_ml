import os
import sys
sys.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from main.LSTNet_v1.utils_v1 import related_trends

import argparse
parser = argparse.ArgumentParser(description = 'Training model')
parser.add_argument('--GT_trends', type=str, default=None,
                    help='Define which Google Trends terms to use: all, related_average, or primary (default)')
args = parser.parse_args()
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
data_cases = pd.read_csv(data_cases_path, usecols=['date_time','num_cases','num_diseased'])
data_cases = data_cases.query(" location == 'Bogotá D.C.-Bogotá d C.' ")
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
data_movement_change['movement_change_7dRA'].iloc[:6] = data_movement_change['movement_change'].iloc[:6]

### Load Google Trends data for Bogota
if args.GT_trends == None:
    data_GT = pd.read_csv(data_GT_path, usecols=['date','anosmia','tos','fiebre','covid'], index_col=1)
elif args.GT_trends == 'all':
    data_GT = pd.read_csv(data_GT_path, index_col=1)
elif args.GT_trends == 'related_average':
    r_select = [0,2,3,4,5,7,10,12,13,16,17,19,24,25]
    data_main_terms   = pd.read_csv(data_GT_id_terms_path)
    data_search_terms = pd.read_csv(data_GT_search_terms_path)
    data_GT = pd.read_csv(data_GT_path, index_col=1)
    data_GT = related_trends(data_search_terms, data_main_terms, data_GT, r_select)