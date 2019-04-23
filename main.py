'''
load dataset
'''

import parser
import os
import sys
from perprocess import Preprosess



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',help = 'Path to .mat files',default = './SAT-4_and_SAT-6_datasets/sat-4-full.mat')
    parser.add_argument('--epochs',help = 'Number of epochs',type = int , default = 25)
    args = parser.parse_args()

    #load dataset
    
