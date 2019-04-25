'''
train different models

'''
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense ,Conv2D , Input,Flatten,Dropout
from tensorflow.keras.applications import ResNet50,MobileNet,Xception , DenseNet121
import ConvNet

def train_model(X_train,Y_train,X_test,Y_test,mode,epochs,model_name = None , model_path = None):
    clear_session()
    # mode = agrs.mode
    # model_name = args.model_name
    if mode == 'Train':
        if model_name == 'ConvNet':
            Convnet = ConvNet()
            Convnet.train(X_train,Y_train,X_test,Y_test,16,args.epochs)
    else:
        model = load_model(model_path)
