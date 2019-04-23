'''
load dataset
split dataset into train and test set
'''

import numpy as np
import scipy.io as sio

class Preprosess(object):
    def __init__(self,datapath):
        self.datapath = datapath

    def load_dataset(self):
        self.data = sio.loadmat(self.datapath)
        self.X_train = self.data['train_x'].transpose()
        self.Y_train = self.data['train_y'].transpose()
        self.X_test = self.data['test_x'].transpose()
        self.Y_test = self.data['test_y'].transpose()
        self.annotation = self.data['annotation']

        self.X_train = self.X_train.reshape(self.X_train.shape[0] , 28, 28, 4)
        self.X_test = self.X_test.reshape(self.X_test,shape[0] , 28, 28, 4)

        return X_train,Y_train,X_test,Y_test

    def visualize_data(self):
        per_class_counts_train = np.sum(self.Y_train,axis = 0)
        per_class_counts_test = np.sum(self.Y_test ,axis = 0)
        print(per_class_counts_train , per_class_counts_test)
