# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:11:26 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

univ=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/clustering/Universities.csv')
univ.isna().sum()

#need to normalize the data
def normalize_func(i):
    x=(i-i.mean())/i.std()
    return (x)
#or using scaling also directly can do normalization
#from sklearn.preprocessing import scale
#norm_data=pd.DataFrame(scale(univ.iloc[:,1:]))


norm_data=normalize_func(univ.iloc[:,1:])
norm_data.describe

########Dendrogram############
from scipy.cluster.hierarchy import linkage
z=linkage(norm_data,method='complete',metric='euclidean')
plt.figure(figsize=(15,5));plt.title('Hierarchial clustering dendogram');plt.xlabel('labels');plt.ylabel('distance')
import scipy.cluster.hierarchy as sch
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=8.)
plt.show()

#################clustering##############
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(norm_data)
type(h_complete)
h_complete.labels_
cluster_labels=pd.DataFrame(h_complete.labels_)
cluster_labels.value_counts()
final_data=pd.concat([cluster_labels,univ],axis=1)
final_data.rename(columns={0:'clusters'},inplace=True)

final_data.groupby(final_data.clusters).mean()

final_data.to_csv('final_univ.csv',index=False)
