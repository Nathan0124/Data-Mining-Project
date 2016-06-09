
# coding: utf-8

# In[1]:

import numpy as np, scipy as sp,pandas as pd, matplotlib.pyplot as plt
import matplotlib, sklearn
import os,sys,csv
import util


# In[2]:

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


import collections
import cPickle as pk

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score


# In[7]:

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

from sklearn.externals import joblib
def train_model(datadir, feaTitles, lst, modeldir, isBooking = False, startGroup=None):
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    
    for i in lst:
        print i
        datapath = os.path.join( datadir,str(i) + '.csv')
        if not os.path.exists(datapath):
            continue
        
        
        df = pd.read_csv(datapath)
        
        util.time_feature_processing(df, True)
        util.nan_feature_processing(df,'orig_destination_distance')
        
        
        skip = True
        for  dst, group in df.groupby('srch_destination_id'):
            print i, dst,
            
            if dst == startGroup:
                    skip = False
            if startGroup!=None and skip:
                    continue       
            
            X = group[feaTitles].as_matrix()
            y = group.hotel_cluster.as_matrix()
            
            weight = np.array(group['is_booking'].tolist())
            if not isBooking:
                weight = 3 * weight + 1
                
            clf = RandomForestClassifier(n_estimators=10, n_jobs=3)
            clf.fit(X, y, weight)
            modelfile = os.path.join(modeldir, str(dst)+'.pkl')
            joblib.dump(clf, modelfile)
            
            print "Done"
            
feaH = ['date_time',
           'in_date',
           'in_days',
           'site_name',
           'posa_continent',
           'user_location_city',
           'orig_destination_distance',
           'user_id',
           'is_mobile',
           'is_package',
           'channel',
           'srch_children_cnt',
           'srch_rm_cnt']
#train_model('../data/byCountry/train',feaH, [182,77,204,70,105,198], '../models/all_data', isBooking = False, startGroup=None)
train_model('../data/byCountry/train',feaH, range(51,70), '../models/all_data', isBooking = False, startGroup=None)