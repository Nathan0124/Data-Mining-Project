
# coding: utf-8

# In[29]:

import numpy as np, scipy as sp,pandas as pd, matplotlib.pyplot as plt
import matplotlib, sklearn
import os,sys,csv
import util


# In[1]:

origin_headers = ['date_time', 
           'site_name',
           'posa_continent',
           'user_location_country',
           'user_location_region',
           'user_location_city',
           'orig_destination_distance',
           'user_id',
           'is_mobile',
           'is_package',
           'channel',
           'srch_ci',
           'srch_co',
           'srch_adults_cnt',
           'srch_children_cnt',
           'srch_rm_cnt',
           'srch_destination_id',
           'srch_destination_type_id',
           'hotel_continent',
           'hotel_country',
           'hotel_market',
           'is_booking',
           'cnt',
           'hotel_cluster']

updated_headers = ['date_time',
           'in_date',
           'in_days',
           'site_name',
           'posa_continent',
           'user_location_country',
           'user_location_region',
           'user_location_city',
           'hotel_continent',
           'hotel_country',
           'hotel_market',
           'srch_destination_id',
           'orig_destination_distance',
           'user_id',
           'is_mobile',
           'is_package',
           'channel',
           'srch_adults_cnt',
           'srch_children_cnt',
           'srch_rm_cnt',
           'srch_destination_type_id',
           'is_booking',
           'cnt',
           'hotel_cluster']


# ## Modeling & Ranking

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score


# In[110]:

import ml_metrics as mtr
def map_5_scorer(estimator, X, y):
    if X.shape[0] == 0:
        return 1
    prob = estimator.predict_proba(X)
    labels = np.array(estimator.classes_)
    
    def top5(prob):
        indice = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
        return labels[indice].tolist()
    
    y = map(lambda x:[x], y)
    y_pred = np.apply_along_axis(top5, axis=1, arr=prob)
    return mtr.mapk(y, y_pred, 5) 


# ### 1. RandomForest

# In[111]:

def cross_validation(df, feaTitles, split = 5, n_est = 10):
    
    X = df[feaTitles].as_matrix()
    y = df.hotel_cluster.as_matrix()
    count = len(df)
    part = int(count / split)

    if X.shape[0] <= split:
        return 1, 0
    elif X.shape[0] <= split * split:
        rfc = RandomForestClassifier(n_estimators=n_est, n_jobs=3)
        #print  X[part:,:], y[part:]
        rfc.fit(X[part:,:], y[part:])
        return map_5_scorer(rfc, X[:part,:], y[:part]), len(df)

    if n_est == 0 or n_est == None:
        n_est = len(feaTitles)
    estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="median",
                                          axis=0)),
                      ("forest", RandomForestClassifier(n_estimators=n_est,
                                                       n_jobs=3))])
    #from util import map_5_scorer
    score = cross_val_score(estimator, X, y, cv=split, scoring=map_5_scorer)
    return score.mean(), len(df)


# In[114]:

def test_model(datadir, feaTitles, isBooking = False, split = 3, dstfile = None):
    scores = []
    freqs = []
    
    if dstfile!=None:
        writer = csv.writer(open(dstfile, 'ab'))
    else:
        writer= None
        
    for i in range(113, 213):
        print i
        datapath = os.path.join( datadir,str(i) + '.csv')
        if not os.path.exists(datapath):
            continue
        
        df = pd.read_csv(datapath)

        util.time_feature_processing(df, True)
        util.nan_feature_processing(df,'orig_destination_distance')
        
        if isBooking:
            df = df[df.is_booking == 1]        

        if len(df) == 0:
        	continue

        for  dst, group in df.groupby('srch_destination_id'):
                
                if len(group) == 0:
                    continue
                score, count = cross_validation(group, feaTitles, split)
                scores.append(score)
                freqs.append(count)
                
                print dst, score, count
                if writer!= None:
                    writer.writerow([str(i), str(dst), str(score), str(count)])
                    
    return np.average(scores, weights=freqs)
            
# In[113]:

fea_header = ['date_time',
           'in_date',
           'in_days',
           'site_name',
           'posa_continent',
           'user_location_country',
           'user_location_region',
           'user_location_city',
           'orig_destination_distance',
           'user_id',
           'is_mobile',
           'is_package',
           'channel',
           'srch_adults_cnt',
           'srch_children_cnt',
           'srch_rm_cnt',
           'srch_destination_type_id']
test_model('../data/byCountry', feaTitles=fea_header,isBooking=True, dstfile='../eval/booking_cv.csv')

