# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:18:43 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

crime_data=pd.read_csv('C:/Users/HP/Desktop/assignments submission/clustering/crime_data.csv')
crime_data.isnull().sum()

def norm_func(i):
    x=(i-i.mean())/i.std()
    return x

norm_data=norm_func(crime_data.iloc[:,1:])

###########dendrogram############
from scipy.cluster.hierarchy import linkage
z=linkage(norm_data,method='complete',metric='euclidean')
plt.figure(figsize=(15,5));plt.xlabel('labels');plt.ylabel('distance')
import scipy.cluster.hierarchy as sch
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=8.)
plt.show()

##############Agglomerative clustering#############
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(norm_data)
type(h_complete)
cluster_labels=pd.DataFrame(h_complete.labels_)
final_data=pd.concat([cluster_labels,crime_data],axis=1)
final_data.rename(columns={0:'clusters'},inplace=True)

final_data.groupby(final_data.clusters).mean()

final_data.to_csv('crime_final.csv')
