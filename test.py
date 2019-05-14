'''
Load trained model and predict Results
'''

import cv2
import argparse
import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocess import Preprosess


def evaluate_model(model, test_images, test_labels):
    loss , acc = model.evaluate(test_images , test_labels)
    return loss, acc
def predict(model , test_images,test_labels):
    #genertae random number
    test_image = test_images[10]
    #show image
    imgplot = plt.imshow(test_image)
    plt.show()

    test_image = test_image.reshape(1,28,28,4)
    test_label = test_labels[10]

    #make prediction
    predictions = np.round(model.predict(test_image))[0]
    #labels
    categories = ['barren_land','trees','grassland','none']
    #predicted label
    labels = [categories[idx] for idx, current_prediction in enumerate(predictions) if current_prediction == 1]
    #actual label
    act_label = [categories[idx] for idx, current_prediction in enumerate(test_label) if current_prediction == 1]
    print("Actual label:{},Predicted label:{}".format(act_label,labels))

def load_model(args):
    #Load model
    model = tf.keras.models.load_model(args.model_path+'Sat_classification.h5')
    #load latest weights
    latest = tf.train.latest_checkpoint(args.model_path)
    model.load_weights(latest)
    model.summary()
    #load test dataset
    Prep = Preprosess(args.data_path)
    test_images , test_labels = Prep.load_testdata()
    if args.mode == 'accuracy':
        loss, acc = accuracy(model,test_images,test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    else:
        predict(model, test_images , test_labels)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",help = "Model path")
    parser.add_argument("--data_path",help = "Path to image",default = './SAT-4_and_SAT-6_datasets/sat-4-full.mat')
    parser.add_argument("--mode",choices=['predict','accuracy'],default = 'predict')
    args = parser.parse_args()
    load_model(args)
