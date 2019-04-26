'''
train different models

'''
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

        elif model_name == 'ResNet50':
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # Create checkpoint callback
            print("checkpoint",checkpoint_path)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1,period=5)
            img = Input(shape = (224,224,4))
            model = ResNet50(include_top=False,
                            weights='imagenet',
                            input_tensor=img,
                            input_shape=None,
                            pooling='avg')
            final_layer = model.layers[-1].output
            dense_layer_1 = Dense(128, activation = 'relu')(final_layer)
            output_layer = Dense(4, activation = 'sigmoid')(dense_layer_1)
            model = Model(input = img, output = output_layer)
            model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])
            model.fit(X_train,Y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_data = (
                                X_test,Y_test),
                        callbacks = [cp_callback])
            #loss graphs
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
        # elif model_name =='De'
    else:
        model = load_model(model_path)
