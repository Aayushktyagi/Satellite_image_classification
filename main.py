'''
load dataset
'''

import parser
import os
import sys
import argparse
import numpy as np
from preprocess import Preprosess


def process(args):
    Prep = Preprosess(args.datapath)
    #load dataset
    X_train , Y_train , X_test , Y_test = Prep.load_dataset()
    print("Dataset : Train:{}:{},Test:{}:{}".format(np.shape(X_train),np.shape(Y_train),np.shape(X_test),np.shape(Y_test)))
    if args.visualize_data == 'True':
        Prep.visualize_data()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',help = 'Path to .mat files',default = './SAT-4_and_SAT-6_datasets/sat-4-full.mat')
    parser.add_argument('--epochs',help = 'Number of epochs',type = int , default = 25)
    parser.add_argument('--visualize_data', help = 'set true to visualize data',default=False)
    parser.add_argument('--mode' , choices = ['train','inference'],default = True)
    args = parser.parse_args()
    process(args)
