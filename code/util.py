
# coding: utf-8

# ## Import Package

import numpy as np, scipy as sp,pandas as pd, matplotlib.pyplot as plt
import matplotlib, sklearn
import os,sys,csv
import ml_metrics as mtr


# ## Data IO
def csv_nrows(fpath):
    count = 0
    with open(fpath, "r") as csvfile:
        reader = csv.reader(csvfile)
        reader.next()
        '''
        for line in reader:
            count += 1
        '''
        count = sum(1 for row in reader)
    return count

def csv_split(fpath, nrows=5000000):
    count = csv_nrows(fpath)
    fname, ext = os.path.splitext(fpath)
    with open(fpath, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = reader.next()
        
        count = 0
        for line in reader:
            if count% nrows == 0:
                print "New Patch", int(count//nrows)
                dpath = fname + '_' + str(count//nrows) + ext
                writer = csv.writer(open(dpath, 'wb'))
                writer.writerow(header)
            writer.writerow(line)
            count += 1

# ## Feature Processing
def time_feature_processing(df,  drop = True):
    def time_interval(row):
        if pd.isnull(row['srch_ci']) or pd.isnull(row['srch_co']):
            return 1
        st = pd.Period(row['srch_ci'],freq='D')
        et = pd.Period(row['srch_co'],freq='D') 
        return et-st
    
    def time_midpoint(row):
        if pd.isnull(row['srch_ci']) or pd.isnull(row['srch_co']):
            return 0
        st = pd.Period(row['srch_ci'],freq='D')
        et = pd.Period(row['srch_co'],freq='D') 

        date = st + (et - st) / 2
        return  date.dayofyear - 1
    
    def time_dayofyear(row):
        if pd.isnull(row['date_time']):
            return 0
        tstamp = pd.to_datetime(row['date_time'])
        return tstamp.dayofyear - 1 + tstamp.hour / 24.
    
    
    df['in_days'] = df.apply(time_interval, axis= 1)
    df['in_date'] = df.apply(time_midpoint, axis= 1)
    df['date_time'] = df.apply(time_dayofyear, axis = 1)
    
    if drop:
        df.drop('srch_ci', axis=1, inplace=True)
        df.drop('srch_co', axis=1, inplace=True)
    else:
        df['srch_ci'] = pd.PeriodIndex(df['srch_ci'], freq='D').dayofyear -1
        df['srch_co'] = pd.PeriodIndex(df['srch_co'], freq='D').dayofyear -1

def nan_feature_processing(df, colname):
    med = df[colname].median()
    df[colname] = df[colname].fillna(med)



# Result Processing & # Evaluating
def uniqify(seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: 
            continue
        seen[marker] = 1
        result.append(item)
    return result


def map_5_scorer(estimator, X, y):
    prob = estimator.predict_proba(X)
    def top5(row):
        return sorted(range(len(row)), key=lambda k: row[k], reverse=True)
    
    y = map(lambda x:[x], y)
    y_pred = np.apply_along_axis(top5, axis=1, arr=prob)
    return mtr.mapk(y, y_pred, 5) 

def mapK(y, y_pred, k):
    return mtr.mapk(y, y_pred, k)  