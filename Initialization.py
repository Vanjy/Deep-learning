#Download the vgg16.h5 and vgg_bn.h5 files from http://files.fast.ai/models/ and import to the .keras/models folder

#Load line by line to avoid using up system memory
%matplotlib inline

#Get the working directory
import sys
print (sys.version) 
import os
print(os.getcwd())

#Create the folder manually in your working directory to insert your test images 
path = "data/dogbreeds/"

#Import some important libraries
from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

#Import Theano and Keras after changing the backend in keras.json file from "tf" to "th"
#Use a text editor ie. Notepad++ to edit the keras.json file
#Use compatible theano and keras versions 
#Install keras using !pip install keras
#Install theano using !pip install theano
# Theano
import theano
print('Theano: %s' % theano.__version__)
import keras
print('Keras: %s' % keras.__version__)

#Import the additional libraries in utils.py file
import utils; reload(utils)
from utils import plots

# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16
vgg = Vgg16()

#Parse some test images to the vgg16 pretrained model
#Note to preprocess images before parsing them
batches = vgg.get_batches(path+'test6', batch_size=1032)
imgs,labels = next(batches)
prediction1 = vgg.predict(imgs, True)

#Store the output of the prediction as a .csv file
import sys
orig_stdout = sys.stdout
f = open(path+'Results\Prediction.csv', 'w',)
sys.stdout = f

import string
for item in prediction1:
  temp = (','.join(str(s) for s in item) + '\n') 
  print(string.replace(temp, '\n', ''))

sys.stdout = orig_stdout
f.close()
