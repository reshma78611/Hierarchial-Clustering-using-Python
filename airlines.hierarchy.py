# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:52:45 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

airlines=pd.read_excel('C:/Users/HP/Desktop/assignments submission/clustering/EastWestAirlines.xlsx',sheet_name='data')
airlines.isna().sum()

def norm_func(i):
    x=(i-i.mean())/i.std()
    return x

norm_data=norm_func(airlines.iloc[:,1:])

#########dendrogram##########
from scipy.cluster.hierarchy import linkage
z=linkage(norm_data,method='complete',metric='euclidean')
plt.figure(figsize=[15,5]);plt.xlabel('labels');plt.ylabel('distances')
import scipy.cluster.hierarchy as sch
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=8.)
plt.show()

#############Agglomerative clustering##########
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=7,linkage='complete',affinity='euclidean').fit(norm_data)
cluster_labels=pd.DataFrame(h_complete.labels_)
cluster_labels.value_counts()
final_data=pd.concat([cluster_labels,airlines],axis=1)
final_data.rename(columns={0:'clusters'},inplace=True)

final_data.groupby(final_data.clusters).mean()

final_data.to_csv('airlines_final.csv')
