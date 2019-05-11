'''
train different models

'''
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense ,Conv2D , Input,Flatten,Dropout
from tensorflow.keras.applications import ResNet50,MobileNet,Xception , DenseNet121
from ConvNet import ConvNet

def train_model(X_train,Y_train,X_test,Y_test,mode,epochs,batch_size,model_name = None,checkpoint_path=None , model_path = None,):
    clear_session()
    # mode = agrs.mode
    # model_name = args.model_name
    if mode == 'train':
        if model_name == 'ConvNet':
            print("Choosen model ConvNet")
            Convnet = ConvNet()
            Convnet.train(X_train,Y_train,X_test,Y_test,batch_size,epochs,checkpoint_path)
            Convnet.showloss()

        else:
            raise ValueError("Select Correct model")
