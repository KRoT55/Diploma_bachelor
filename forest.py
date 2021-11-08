import pandas as pd
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import scipy
import skimage
from skimage.transform import resize
from keras import backend as K
import glob
import time
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
K.common.image_dim_ordering()


start_time = time.time()
#X_train=np.load("X_train.npy")
y_train = np.load("y_train.npy")
#X_test=np.load("X_test.npy")
y_test = np.load("y_test.npy")
X_val=np.load("X_val.npy")
y_val = np.load("y_val.npy")

tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)
#c=X_train.shape
#nsamples, nx, ny,a = X_train.shape

#X_train = X_train.reshape((nsamples,nx*ny*a))

#nsamples2, nx2, ny2,a2 = X_test.shape
#X_test = X_test.reshape((nsamples2,nx2*ny2*a2))



#end_time =time.time()-start_time
#print("end reshape :", end_time)


#np.save("X_train-forest",X_train)
#np.save("y_train",y_train)
#np.save("X_test-forest",X_test)
#np.save("y_test",y_test)
#np.save("X_val-forest",X_val)
#np.save("y_val",y_val)
X_train=np.load("D:\диплом\CNN\X_train-forest.npy")
X_test=np.load("D:\диплом\CNN\X_test-forest.npy")

tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
acc=accuracy_score(y_test, tree_pred) # 0.94
acc_str=acc
acc_str=str(acc_str)
print("1acc tree : ", acc_str )
end_time1 =time.time()-start_time
print("end 1tree :", end_time1)

start_time1 = time.time()
knn_pred = knn.predict(X_test)
bacc=accuracy_score(y_test, knn_pred) # 0.88
print("2acc knn : ", bacc)
end_time =time.time()-start_time1
print("end 2knn :", end_time)

plt.plot(acc,y_test)
#plt.plot(end_time1['time'])
plt.title('model forest accuracy')
plt.ylabel('time')
plt.xlabel('accuracy')
plt.show()
# summarize history for loss
plt.plot(bacc,y_train)
plt.title('nearest neighbor')
plt.ylabel('time')
plt.xlabel('accuracy')
plt.show()

#tree = DecisionTreeClassifier(random_state=17, max_depth=1)
#tree_cv_score = np.mean(cross_val_score(tree, X_train, y_train, cv=5))
#tree.fit(X_train, y_train)
#tree_holdout_score = accuracy_score(y_holdout, tree.predict(X_test))
#print('Decision tree. CV: {}, holdout: {}'.format(tree_cv_score, tree_holdout_score))

data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
names = list(data.keys())
values = list(data.values())
a= 0.62
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].bar((acc,bacc,a),1)
fig.title('Classic methods')