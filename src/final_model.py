''' This code is used to classify images with traffic signs using Fast RCNN approach'''

#Global Variables Used in program
IMG_SIZE = 480
NUM_CLASSES = 3
batch_size = 8
file_name = 'final_model_reg_3_w_img480_chmset_val_w_aug'
precision_file_name_json = "/home/ubuntu/scores/" + file_name +".json"
model_save_file_name = "/home/ubuntu/saved_models/" + file_name + ".h5"
loss_save_file_name = '/home/ubuntu/loss_history/' + file_name +'.json'


import tensorflow as tf
sess = tf.Session()
import json
import random
import cPickle as pickle
import keras
from keras.utils import to_categorical
from keras.layers import Input
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn.metrics import average_precision_score, precision_score, recall_score
from test_fns import testing_results, test_images
from utils import convert_tocat, msetf, get_data, image_names, data_splitter
from augment import augment_image, reize_image, generator
from cnn_arch import compile_model, create_model
sess = tf.Session()


if __name__=='__main__':
    #Creating Training and Validation generatore
    df = get_data()
    train_fn = image_names('/home/ubuntu/train/')
    train_df = data_splitter(train_fn, df)
    train_generator = generator('/home/ubuntu/train/', train_df, (IMG_SIZE,IMG_SIZE), 8)
    val_fn = image_names('/home/ubuntu/val/')
    val_df = data_splitter(val_fn, df)
    val_generator = generator('/home/ubuntu/val/', val_df, (IMG_SIZE, IMG_SIZE), 8)

    #Creating, compiling and running model
    model = compile_model()
    fit_model = model.fit_generator(generator=train_generator,validation_data = val_generator, validation_steps = 10, steps_per_epoch=100, epochs=100)
    model.save(model_save_file_name)

    #Writing Loss to file
    with open(loss_save_file_name, 'w') as f1:
        json.dump(model.history.history,f1)
    f1.close()
    
    #Writing test and results to file 
    testing_result(df, model)
    
