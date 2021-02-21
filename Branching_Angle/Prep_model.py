'''
Created on 10 Jan 2021

@author: danan
'''

import numpy as np
import ONH_Localization as ONHLOC
import ONH_Segmentation as ONHSEG
from sklearn.svm import LinearSVC
from joblib import dump, load

from skimage import io


# train model ONH

train_txt= 'feature_mat_norm_all.csv'
label_txt= 'Labels_vec_all.csv'
feature_mat_norm_all = np.loadtxt(train_txt, delimiter=',')
Labels_vec_all = np.loadtxt(label_txt, delimiter=',')
clf = LinearSVC(C=1, max_iter=10000)
model = clf.fit(feature_mat_norm_all, Labels_vec_all)

dump(model, 'ONH_model.joblib') 
