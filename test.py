'''
Load trained model and predict Results
'''

import cv2
import argparse
import numpy
import tensorflow as tf


def evaluate_model(model, test_images, test_labels):
    loss , acc = model.evaluate(test_images , test_labels)
    return loss, acc
def predict(model , test_images):
    predictions = model.predict(test_images)
    # print('predictions shape:', predictions.shape)
    return  predictions

def load_model(args):
    #Load model
    model = keras.models.load_model(args.model_path+'Sat_classification.h5')
    #load latest weights
    latest = tf.train.latest_checkpoint(args.model_path)
    model.load_weights(latest)
    if args.mode = 'accuracy':
        loss, acc = accuracy(model,test_images,test_labels)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    else:
        predictions = predict(model, test_images)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",help = "Model path")
    parser.add_argument("--image_path",help = "Path to image")
    parser.add_argument("--mode",choices=['predict','accuracy'],default = 'predict')
    args = parser.parse_args()
    load_model(args)
