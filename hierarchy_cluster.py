import torch
import h5py
from sklearn.cluster import MiniBatchKMeans
f1 = h5py.File('data/MSRVTT/msrvtt_concept1000_feat_train.h5','r')
print(f1['concept_features'].shape)

kmeans = MiniBatchKMeans(n_clusters=500, random_state=0, batch_size = 1024).fit(f1['concept_features'])

f = h5py.File('msrvtt_concept500_feat_train.h5','w')  
f['concept_features'] = kmeans.cluster_centers_                         
f.close()