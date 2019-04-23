'''
load dataset
split dataset into train and test set
'''
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class Preprosess(object):
    def __init__(self,datapath):
        self.datapath = datapath

    def load_dataset(self):
        self.data = sio.loadmat(self.datapath)
        self.X_train = self.data['train_x'].transpose()
        self.Y_train = self.data['train_y'].transpose()
        self.X_test = self.data['test_x'].transpose()
        self.Y_test = self.data['test_y'].transpose()
        self.annotation = self.data['annotations']
        print(self.annotation)

        self.X_train = self.X_train.reshape(self.X_train.shape[0] , 28, 28, 4)
        self.X_test = self.X_test.reshape(self.X_test.shape[0] , 28, 28, 4)

        return self.X_train,self.Y_train,self.X_test,self.Y_test

    def visualize_data(self):
        if os.path.splitext(os.path.basename(self.datapath))[0] =='sat-4-full':
            labels = ['barren land' , 'trees' , 'grassland','others']
        else:
            labels =['building','barren land','trees','grassland','roads','water']
        per_class_counts_train = np.sum(self.Y_train,axis = 0)
        per_class_counts_test = np.sum(self.Y_test ,axis = 0)
        index = np.arange(len(per_class_counts_train))
        plt.bar(index,per_class_counts_train)
        plt.xlabel("Labels",fontsize = 10)
        plt.ylabel("Count" , fontsize = 10)
        plt.xticks(index , labels , fontsize=10,rotation = 30)
        plt.title("Class wise distribution for train data")
        plt.show()

        #For test data
        index = np.arange(len(per_class_counts_test))
        plt.bar(index,per_class_counts_test)
        plt.xlabel("Labels",fontsize = 10)
        plt.ylabel("Count" , fontsize = 10)
        plt.xticks(index , labels , fontsize=10,rotation = 30)
        plt.title("Class wise distribution for test data")
        plt.show()
