
# coding: utf-8

# In[1]:

import numpy as np, scipy as sp,pandas as pd, matplotlib.pyplot as plt
import matplotlib, sklearn
import os,sys,csv
import util
import collections
import cPickle as pk

def geo_stat(fpath, isBooking = True):
    fname, ext = os.path.splitext(fpath)
    with open(fpath, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = reader.next()
        book_item = header.index('is_booking')
        destination_item = header.index('srch_destination_id')
        country_item = header.index('hotel_country')
        hotel_item = header.index('hotel_cluster')
        
        dest_map = {}
        country_map = {}
        
        count = 0
        for line in reader:
            count += 1
            if count%1000000 == 0:
                print "Processed:", str(count)
            
            if isBooking and line[book_item] != '1':
                continue
                
            destination_id = int(line[destination_item])
            country_id = int(line[country_item])
            hotel_cluster = int(line[hotel_item])
            
            value1 = dest_map.setdefault(destination_id, [])
            value1.append(hotel_cluster)
            
            value2 = country_map.setdefault(country_id, [])
            value2.append(hotel_cluster)          
            
    print 'IO Done. Now Processing...'
    
    for k, v in dest_map.iteritems():
        v.sort(key=collections.Counter(v).get, reverse=True)
        dest_map[k] = uniqify(v)
    
    for k, v in country_map.iteritems():
        v.sort(key=collections.Counter(v).get, reverse=True)
        country_map[k] = uniqify(v)
        
    print 'Done'
    return dest_map, country_map


def save_geo_stat(dstpath, dst_map, cntry_map):
    with open(dstpath,'wb') as fp:
        pk.dump(dst_map ,fp)
        pk.dump(cntry_map,fp)
        
def load_geo_stat(srcpath):
    with open(srcpath,'rb') as fp:
        dst_map = pk.load(fp)
        cntry_map = pk.load(fp)
        return dst_map, cntry_map
    

def csv_map_by_key(srcpath, dstdir, key='hotel_country'):
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
        
    data = pd.read_csv(srcpath)
    for name, group in data.groupby(key):
        filepath = os.path.join(dstdir, str(name) + '.csv')
        skipHeader = os.path.exists(filepath) 
        with open(filepath, "a") as csvfile:
            group.to_csv(csvfile, mode='a', header=(not skipHeader))
            
def loop_map_by_key(dstdir):
    #fname, ext = os.path.splitext(fpath)
    for i in range(8):
        print i
        datapath = '../data/train_' + str(i) + '.csv'
        csv_map_by_key(datapath, dstdir)
        


# In[ ]:

#loop_map_by_key('../data/byCountry')
csv_map_by_key('../data/booking_train.csv', '../data/byDesitination', key='srch_destination_id')



