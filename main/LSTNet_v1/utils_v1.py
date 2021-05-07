import itertools
import numpy as np
import pandas as pd

def related_trends(search_terms, main_terms, trends, r_list):
    df_related_trends_avg = []
    for r in r_list:
        # Get related trend
        _, related_trend_avg, trend_name = get_related_trend(search_terms, main_terms, trends, r)
        # save
        df_trend = pd.DataFrame()
        df_trend['date'] = related_trend_avg['date']
        df_trend[trend_name] = related_trend_avg['weighted_avg']
        df_trend = df_trend.set_index('date')
        df_related_trends_avg.append(df_trend)
    df_related_trends_avg = pd.concat(df_related_trends_avg)
    return df_related_trends_avg


def get_related_trend(search_terms, main_terms, trends, r_in):
    entry_terms = search_terms.columns.values[1:]
    map_term_2_r = dict(zip(search_terms['term'], search_terms['r']))
    map_r_2_term = dict(zip(main_terms['r'], main_terms['id']))
    # Get related terms
    df_related_terms = []
    maximuns = []
    for kw in entry_terms:
        if r_in == map_term_2_r[kw]:
            kw_trend = trends[kw]; kw_trend = kw_trend.reset_index()
            kw_trend["date"] = pd.to_datetime( kw_trend["date"] ) 
            kw_trend['rw']   = kw_trend[kw].rolling(window=7).mean()
            kw_trend['term'] = [kw]*len(kw_trend)
            kw_trend = kw_trend.drop(columns=[kw])
            df_related_terms.append(kw_trend)
            maximuns.append( kw_trend['rw'].max()/100 )
    df_related_terms = pd.concat( df_related_terms )
    df_weighted_avg  = get_weighted_average(df_related_terms, maximuns)
    df_weighted_avg_res = pd.DataFrame()
    df_weighted_avg_res['date'] = df_weighted_avg['date']
    df_weighted_avg_res['weighted_avg'] = df_weighted_avg[0]
    main_trend_name = map_r_2_term[r]
    return df_related_terms, df_weighted_avg_res, main_trend_name


def get_weighted_average(df_entry, maximuns):
    df_mean = df_entry.copy()
    df_mean = df_mean.dropna()
    df_mean["weights"] = list(itertools.chain(*[[w]*(len(df_mean)//len(maximuns)) for w in maximuns]))  
    df_wa = df_mean.groupby(['date']).apply(lambda x: pd.Series([ np.average(x['rw'], weights=x['weights']) ]) ).reset_index()
    return df_wa