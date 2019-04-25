'''
load dataset
train model
check classification metrics for classification
'''

import parser
import os
import sys
import argparse
import numpy as np
from preprocess import Preprosess
import model

def process(args):
    Prep = Preprosess(args.datapath)
    #load dataset
    X_train , Y_train , X_test , Y_test = Prep.load_dataset()
    print("Dataset : Train:{}:{},Test:{}:{}".format(np.shape(X_train),np.shape(Y_train),np.shape(X_test),np.shape(Y_test)))
    if args.visualize_data == 'True':
        Prep.visualize_data()
    model.train_model(X_train,Y_train,X_test,Y_test,args.mode,args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',help = 'Path to .mat files',default = './SAT-4_and_SAT-6_datasets/sat-4-full.mat')
    parser.add_argument('--epochs',help = 'Number of epochs',type = int , default = 25)
    parser.add_argument('--visualize_data', help = 'set true to visualize data',default=False)
    parser.add_argument('--mode' , choices = ['train','inference'],default = 'train')
    parser.add_argument('--output' , help= 'path checkpoints will be saved',default='./weights/')
    parser.add_argument('--model_name',choices=['ConvNet','ResNet50','Xception','MobileNet','DenseNet121'],default='ConvNet')
    args = parser.parse_args()
    process(args)
