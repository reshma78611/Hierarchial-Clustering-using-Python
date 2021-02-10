# Hierarchial-Clustering-using-Python

In **Unsupervised Learning** we have different type of algorithms such as:
1. Clustering
2. Association Rules
3. Recommendation Engine
4. PCA
5. Text mining
6. NLP


In **Clustering** we have :
1. Hierarchial Clustering
2. K-Means Clustering
3. DBSCAN Clustering

In this repository we will discuss mainly about *Hierarchial Clustering*

 This is mainly used for Numerical data, it is also called as bottom-up approach. In this, among all the records two records which are having less Euclidean distance are merged in to one cluster and again this cluster inturn with which it is having less euclidean distance that record is merged with the cluster.
 
 Now how to calculate distance between 2 clusters or distance between cluster and record ?
 
    For this we have different methods:
    1. Single linkage method : *It will consider minimum distance*
    2. Complete linkage method : *It will consider maximum distance*
    3. Average linkage method : *It will consider average of all distances*
    4. centroid linkage method : *It will consider centroid of records of one cluster and centroid of records of other cluster and now distance between these two centroids is considered.*
    
Summarization of entire cluster process is done using **Dendogram**

**Advantages :**
1. Best suitable for smaller datasets.
2. Dendogram gives best understanding of clustered data.

**Diasadvantages :**
1. It is slower for large datasets.
    

## Data used:
          universities : Clustering of similar universities,
          crime_data : Clustering of Murderers based on their similarity ,
          EastWestAirlines : Clustering of Airlines based on their similarity


##  Programming:
           Python


**The Codes regarding this Hierarchial Clustering with three different business problems *Clustering of uiversities* ,*Clustering of murderers*, *Clustering of Airlines* with their datasets are present in this Repository in detail**
