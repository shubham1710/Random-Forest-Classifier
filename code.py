import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

a=np.genfromtxt('train_X_rf.csv',delimiter=',', skip_header=1)
b=np.genfromtxt('train_Y_rf.csv',delimiter=',')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
clf = RandomForestClassifier(max_depth=18, random_state=45)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
ro=np.shape(x_test)[0]
pred.resize(ro,1)
acc = metrics.accuracy_score(y_test, pred))
print("Accuracy =",acc)
np.savetxt("predicted_test_Y_rf.csv", pred, delimiter=",")
