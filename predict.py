#!/usr/bin/env python
# coding: utf-8

# In[88]:


import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
def predict(test_X_file_path):
    a1=np.genfromtxt(test_X_file_path, delimiter=',', skip_header=1)
    a=np.genfromtxt('train_X_rf.csv',delimiter=',', skip_header=1)
    b=np.genfromtxt('train_Y_rf.csv',delimiter=',')
    clf = RandomForestClassifier(max_depth=18, random_state=45)
    clf.fit(a,b)
    pred = clf.predict(a1)
    ro=np.shape(a1)[0]
    pred.resize(ro,1)
    np.savetxt("predicted_test_Y_rf.csv", pred, delimiter=",")

if __name__ == "__main__":
    predict(sys.argv[1])

