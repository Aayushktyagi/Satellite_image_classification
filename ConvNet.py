'''
convolutional neural network
'''

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input , Dense,Conv2D,MaxPooling2D,Dropout
import matplotlib.pyplot as plt


class ConvNet(object):
    def __init__(self):
        input_layer = Input(shape=(28,28,4))
        layer_1 = Conv2D(8,(2,2),activation = 'relu',padding = 'same')(input_layer)
        layer_1 = MaxPooling2D((2,2),padding ='same')(layer_1)
        layer_1 = Dropout(0.3)
        layer_2 = Conv2D(16,(2,2),activation = 'relu',paddig = 'same')(layer_2)
        layer_2 = MaxPooling2D((2,2),padding = 'same')(layer_2)
        layer_2 = Dropout(0.3)
        layer_3 = Conv2D(32,(2,2),activation = 'relu',padding= 'same')(layer_3)
        layer_3 = MaxPooling2D((2,2),padding = 'same')(layer_3)
        layer_3 = Dropout(0.3)
        layer_flatten = Flatten()(layer_3)
        layer_4 = Dense(784, activation ='relu')(layer_3)
        layer_4 = Dropout(0.3)
        output_layer = Dense(10,activation='softmax')(layer_4)

        self._model = Model(input_layer,output_layer)
        self._model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['accuracy'])
        self._model.summary()

    def train(self, X_train,Y_train,X_test,Y_test,batchsize,epochs,checkpoint_path):

        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1,period=10)
        self._model.fit(X_train,Y_train,
                        batchsize = batchsize,
                        epochs = epochs,
                        validation_data = (X_test,
                                            Y_test),
                        callbacks = [cp_callback])

    def showloss(self):
        # loss plot
        plt.subplot(2,1,1)
        plt.plot(self._model.history.history['loss'])
        plt.plot(self._model.history.hostory['val_loss'])
        plt.title('Model loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Train','Validation'],loc = 'upper left')

        #accuracy plot
        plt.subplot(2,1,2)
        plt.plot(self._model.history.history['acc'])
        plt.plot(self._model.history.history['val_acc'])
        plt.title('Accuracy')
        plt.xlable('epochs')
        plt.ylabel('accuracy')
        plt.legend(['Train','Validation'] , loc = 'upper left')
        plt.show()
