# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:14:34 2023

@author: chaub
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import random as rd
import re
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer


###############################################################################          

#importing data from csv file written in dataprep.py file

dat = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_data_clean.csv", index_col= False)
dat.head()

cols_to_normalize = ['Av_temperature','Percentage_pop_with_bachelors', 'Adults 19-25', 'Adults 26-34', 'Adults 35-54', 'EnergyC_addunit',
'num_jobs_tot', 'Median_income', 'GDP', 'Elec_price',
'Tot_vehicle']

for i in cols_to_normalize:
    print(f' The range of {i} is {dat[i].min()} to {dat[i].max()}')



# For clustering it is important that the data is normalized
# The range for the features varies tremendously
# In this file both original and normalized data will be clustered


    
def normalize(some_list):
    new_list = []
    for i in some_list:
        j = (i-min(some_list))/(max(some_list)-min(some_list))
        new_list.append(j)
    return new_list

example = [3,5,22,34,65,7,8,76,3]
print(normalize(example))

#creating a new list to store the names of the newely created normalized columns
new_name_list = []
# Normalizing columns
for i in cols_to_normalize:
    new_name = 'norm_'+i
    new_name_list.append(new_name)
    dat[new_name] = dat[[i]].apply(normalize)
    
for i in new_name_list:
    print(f' The range of {i} is {dat[i].min()} to {dat[i].max()}')

    
dat.head()
dat.columns
dat.to_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5622 Machine Learning/CSCI5662 Project/EV_data_norm.csv', index = False)
dat['norm_Elec_price']

#Removing rows with year = 2021
print(dat.shape)

#checking the class of Year column
print(dat.info())
# It's a number, so we need to change it to a string
dat['Year'] = dat['Year'].astype(str)


# Plotting before and after normalization

sns.histplot(dat['EnergyC_addunit'])
sns.histplot(dat['norm_EnergyC_addunit'])

sns.histplot(dat['num_jobs_tot'])
sns.histplot(dat['norm_num_jobs_tot'])


sns.histplot(dat['GDP'])
sns.histplot(dat['norm_GDP'])

sns.histplot(dat['Tot_vehicle'])
sns.histplot(dat['norm_Tot_vehicle'])

# STandardizing function to use later if necessary
def standardize(some_list):
    new_list = []
    m = sum(some_list)/len(some_list)
    std = np.std(some_list)
    for i in some_list:
        j = (i-m)/std
        new_list.append(j)
    return new_list

dat.columns 
dat.head(0)

Labels = dat[['Number of registrations', 'Label', 'Label_py']]
print(type(Labels))




#Creating a new dataframe without labels, state name and Year, only keeping the normalized variables       

data_wo_label = dat[['norm_Av_temperature',
       'norm_Percentage_pop_with_bachelors', 'norm_Adults 19-25',
       'norm_Adults 26-34', 'norm_Adults 35-54', 'norm_EnergyC_addunit',
       'norm_num_jobs_tot', 'norm_Median_income', 'norm_GDP',
       'norm_Elec_price', 'norm_Tot_vehicle']]

data_wo_label
data_wo_label.isna().sum()

#the missing values of the temperature need to be fixed
# We will do this in the EDA_EV.py file instead

## Data is ready to be used for k means clustering

#####################
#creating 1st instance of KMeans
# code credit: Dr. Ami Gates

EV_KMean = KMeans(n_clusters = 3)
EV_KMean.fit(data_wo_label)
pred_labels = EV_KMean.predict(data_wo_label)
print(pred_labels)


###############################################################################
#Creating second instance of K means using PCA to reduce dimantionality
pca = PCA(2)
dat_pca = pca.fit_transform(data_wo_label)
dat_pca.shape
print(dat_pca)

EV_KMean_pca = KMeans(n_clusters=3)
EV_KMean_pca.fit(dat_pca)
pred_labels_pca = EV_KMean_pca.predict(dat_pca)
print(pred_labels_pca)

filtered_label_0 = dat_pca[pred_labels_pca == 0]
filtered_label_1 = dat_pca[pred_labels_pca == 1]
filtered_label_2 = dat_pca[pred_labels_pca == 2]

print(filtered_label_1)


plt.scatter(filtered_label_0[:,0], filtered_label_0[:,1], color = 'red')
plt.scatter(filtered_label_1[:,0], filtered_label_1[:,1], color = 'black')
plt.scatter(filtered_label_2[:,0], filtered_label_2[:,1], color = 'green')

plt.show()

###############################################################################

#Performing PCA and visualizing results

pca = PCA()
pca.fit(data_wo_label)

pca.explained_variance_ratio_

#Visualizing the results of PCA
plt.figure(figsize = (10,8))
plt.plot(range(1, len(new_name_list)+1), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.xticks(np.arange(0,11, step = 1))
plt.title('Explained Variance by components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

scores_pca = pca.transform(data_wo_label)

#Rule of thumb is to preserve 80% of variance, so let's keep the first two components.

pca2 = PCA(2)
pca2.fit(data_wo_label)


#############################################################################
#Now let's determine the number of clusters using silhouette and elbow method

k_3 = KMeans(n_clusters=3, random_state=120)
k_3.fit_predict(data_wo_label)
score_2 = silhouette_score(data_wo_label, dat[['Label_py']], metric = 'euclidean')
print(score_2)



fig, ax = plt.subplots(2,2,figsize = (15, 8))

for i in [2,3,4,5]:
    
    km = KMeans(n_clusters=i, init = 'k-means++', n_init = 10, max_iter = 100, random_state=120)
    q, mod = divmod(i, 2)
    
    #creating the visualization
    visualizer = SilhouetteVisualizer(km,colors='yellowbrick', ax = ax[q-1][mod])
    visualizer.fit(data_wo_label)
    
#########ELBOW
print(data_wo_label.shape[1])
wcss= []
for i in range(1, data_wo_label.shape[1]):
    kmeans_pca = KMeans(n_clusters  =i, init = 'k-means++', random_state = 120)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)
    
plt.figure(figsize = (10,8))
plt.plot(range(1,data_wo_label.shape[1]), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('K-Means with PCA clustering')
plt.show()


#######IMPLEMENT
kmean_4 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 120)
kmeans_pca.fit(scores_pca)

######## ADDING PCA RESULTS TO DATAFRAME
pca3 = PCA(n_components = 3)
pca3.fit(data_wo_label)
pca3.transform(data_wo_label)
scores_pca3 = pca3.transform(data_wo_label)
scores_pca3

kmeans4_pca3 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 120)
kmeans4_pca3.fit(scores_pca3)

pca_kmeans_data = pd.concat([data_wo_label.reset_index(drop = True), pd.DataFrame(scores_pca3)], axis = 1)
pca_kmeans_data.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
pca_kmeans_data['PCA label'] = kmeans4_pca3.labels_

pca_kmeans_data.head()


x_axis = pca_kmeans_data['Component 2']
y_axis = pca_kmeans_data['Component 1']
plt.figure(figsize = (10,0))
sns.relplot(x = x_axis, y = y_axis, hue=pca_kmeans_data['PCA label'])
plt.title('Clusters by PCA Components')
plt.show()


#k=3
pca2 = PCA(n_components = 2)
pca2.fit(data_wo_label)
pca2.transform(data_wo_label)
scores_pca2 = pca2.transform(data_wo_label)
scores_pca2

kmeans3_pca2 = KMeans(n_clusters = 3, init = 'k-means++', random_state = 120)
kmeans3_pca2.fit(scores_pca2)

pca_kmeans_data_k2 = pd.concat([data_wo_label.reset_index(drop = True), pd.DataFrame(scores_pca2)], axis = 1)
pca_kmeans_data_k2.columns.values[-2:] = ['Component 1', 'Component 2']
pca_kmeans_data_k2['PCA label'] = kmeans3_pca2.labels_

pca_kmeans_data_k2.head()


x_axis_k3p2 = pca_kmeans_data_k2['Component 2']
y_axis_k3p2 = pca_kmeans_data_k2['Component 1']
plt.figure(figsize = (10,0))
sns.relplot(x = x_axis_k3p2, y = y_axis_k3p2, hue=pca_kmeans_data_k2['PCA label'])
plt.title('Clusters by PCA Components')
plt.show()


#k=5
pca2 = PCA(n_components = 2)
pca2.fit(data_wo_label)
pca2.transform(data_wo_label)
scores_pca2 = pca2.transform(data_wo_label)
scores_pca2

kmeans5_pca2 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 120)
kmeans5_pca2.fit(scores_pca2)

pca_kmeans_data_k5 = pd.concat([data_wo_label.reset_index(drop = True), pd.DataFrame(scores_pca2)], axis = 1)
pca_kmeans_data_k5.columns.values[-2:] = ['Component 1', 'Component 2']
pca_kmeans_data_k5['PCA label'] = kmeans5_pca2.labels_

pca_kmeans_data_k5.head()


x_axis_k5p2 = pca_kmeans_data_k5['Component 2']
y_axis_k5p2 = pca_kmeans_data_k5['Component 1']
plt.figure(figsize = (10,0))
sns.relplot(x = x_axis_k5p2, y = y_axis_k5p2, hue=pca_kmeans_data_k5['PCA label'])
plt.title('Clusters by PCA Components')
plt.show()


####################################################################

#DBSCAN

db_cluster = DBSCAN(eps = 0.3, min_samples = 5).fit(data_wo_label)
data_dup = data_wo_label.copy()
data_dup.loc[:, 'Cluster'] = db_cluster.labels_
print(data_dup['Cluster'].unique())
print(data_dup.Cluster.value_counts().to_frame())



outliers = data_dup[data_dup['Cluster']== -1]

fig2, (axes) = plt.subplots(1,2, figsize = (12,5))

data_sup.columns
sns.scatterplot('norm_Percentage_pop_with_bachelors','norm_Adults 19-25', data = data_dup )



















