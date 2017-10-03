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
# import pdb
import pandas as pd
import numpy as np
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn.metrics import average_precision_score, precision_score, recall_score

sess = tf.Session()

def convert_tocat(test_pred):
    test_pred_cat = []
    for i in test_pred[0]:
        if np.argmax(i) == 0:
            test_pred_cat.append([1,0,0])
        elif np.argmax(i) == 1:
            test_pred_cat.append([0,1,0])
        else:
            test_pred_cat.append([0,0,1])
    return test_pred_cat

def testing_result(df, model):
    test_fn = image_names('/home/ubuntu/test/')
    test_df = data_splitter(test_fn, df)
    test_imgs, test_labels = test_images('/home/ubuntu/test/', test_df, (IMG_SIZE, IMG_SIZE), 8)
    test_pred = model.predict(x = test_imgs)
    test_pred_cat = convert_tocat(test_pred)

    #Calculating Individual Precision
    a = np.array(test_pred_cat).T
    b = np.array(test_labels).T
    # precision_score(y_true=b[0], y_pred=a[0])
    prec_score = []
    rec_score = []
    for i in xrange(len(a)):
        prec_score.append(precision_score(y_true=b[i], y_pred=a[i]))
        print('Precision for class {} is {}'.format(i, prec_score[i]))
        rec_score.append(recall_score(y_true=b[i], y_pred=a[i]))
        print('Recall for class {} is {}'.format(i, rec_score[i]))
    with open(precision_file_name_json, 'w') as f:
        json.dump({'precision':prec_score, 'recall':rec_score}, f)
    f.close()

def msetf(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)/25000.0

def create_model():
    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 3))

    #Block 1
#     layer_1 = Conv2D(32,(3,3), padding = 'same', input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu')
    layer_1 = Conv2D(32,(3,3), padding = 'same', activation = 'relu')(inputs)
    layer_2 = Conv2D(32,(3,3), padding = 'same', activation = 'relu')(layer_1)
    layer_3 = MaxPooling2D(pool_size = (2,2))(layer_2)
#     layer_d1 = Dropout(0.3)(layer_3)

    #Block 2
    layer_4 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(layer_3)
    layer_5 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(layer_4)
    layer_6 = MaxPooling2D(pool_size = (2,2))(layer_5)
#     layer_d2 = Dropout(0.3)(layer_6)

    #Block 3
    layer_7 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(layer_6)
    layer_8 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(layer_7)
    layer_9 = MaxPooling2D(pool_size = (2,2))(layer_8)
#     layer_d3 = Dropout(0.3)(layer_9)

#     #Block 4
#     layer_10 = Conv2D(256,(3,3), padding= 'same', activation = 'relu')(layer_9)
#     layer_11 = Conv2D(256, (3,3), padding = 'same', activation = 'relu')(layer_10)
#     layer_12 = MaxPooling2D(pool_size = (2,2))(layer_11)
# #     layer_d4 = Dropout(0.3)(layer_12)

#     #Block 5 
#     layer_13 = Conv2D(512,(3,3), padding = 'same', activation = 'relu')(layer_12)
#     layer_14 = Conv2D(512,(3,3), padding = 'same', activation = 'relu')(layer_13)
#     layer_15 = MaxPooling2D(pool_size = (2,2))(layer_14)
    
#     #Block 6
#     layer_16 = Conv2D(1024,(3,3), padding = 'same', activation = 'relu')(layer_15)
#     layer_17 = Conv2D(1024,(3,3), padding = 'same', activation = 'relu')(layer_16)
#     layer_18 = MaxPooling2D(pool_size = (2,2))(layer_17)

    #Final Block
    layer_19 = Flatten()(layer_9)
    layer_20 = Dense(512, activation = 'relu')(layer_19)
    softmax_class = Dense(NUM_CLASSES, activation = 'softmax', name= 'softmax_class')(layer_20)
    relu_bbox = Dense(4, activation = 'relu', name = 'relu_bbox')(layer_20)

    model = Model(inputs= inputs, outputs = [softmax_class, relu_bbox])
    return model

def compile_model():
    model = create_model()
    adam = Adam(lr = 1e-8, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay=0.01)
#     model.compile(loss={'relu_bbox':bb_intersection_over_union, 'softmax_class':'categorical_crossentropy'}, optimizer=adam, metrics = ['categorical_accuracy'])
    model.compile(loss={'relu_bbox':msetf, 'softmax_class':'categorical_crossentropy'}, optimizer=adam, metrics = ['categorical_accuracy'])
    return model

def get_data():
    df = pd.read_csv('allAnnotations.csv', sep=';')
    df.drop_duplicates(subset='Filename', inplace = True)
    df = df.loc[df['Annotation tag'].isin(['stop', 'pedestrianCrossing', 'speedLimitUrdbl','speedLimit25','speedLimit35','speedLimit45','speedLimit15','speedLimit40','speedLimit50','speedLimit55','speedLimit30','speedLimit65'])]
    df['Filename'] = [i.split('/')[2] for i in df['Filename']]
    df.drop(['Occluded,On another road', 'Origin frame number', 'Origin track','Origin track frame number','Origin file'], axis = 1, inplace = True)
    df['Annotation tag'] = [i.split('_')[0] for i in df['Filename']]
    df['Annotation tag'].replace(to_replace = ['pedestrian'], value = 'pedestrianCrossing',inplace = True)
    df['Annotation tag'].replace(to_replace = ['speedLimitUrdbl','speedLimit25','speedLimit35','speedLimit45','speedLimit15','speedLimit40','speedLimit50','speedLimit55','speedLimit30','speedLimit65'], value='speedLimit', inplace= True)
    return df

def image_names(path):
    ''' 
    Gets the file names in given path. Should enter the path for Train/Test/Validation
    All classes must be in sub directories inside the path. 
    Input:
        path = The path for data (can be train/test/validation)
    
    Output:
        filename = List of all files in a list of all classes

    '''
    filename = []
#     for classes in listdir('/Users/praveen/Downloads/signDatabase/train/')[1:]:
    for classes in listdir(path)[:3]: 
#         filename.append(listdir('/Users/praveen/Downloads/signDatabase/train/'+str(classes)+'/'))
        filename.append(listdir(path+str(classes)+'/'))
    return filename

def data_splitter(list_names,df):
    new_df = df.loc[df['Filename'].isin(list_names[0]) | df['Filename'].isin(list_names[1]) | df['Filename'].isin(list_names[2])] 
    new_df.reset_index(inplace=True)
    return new_df

#1 - move right, down  and   -1 - move left, up
def augment_image(img, bbox):
    width, height = img.size
    rand_x = int((random.uniform(0,1.5)/10.0)*width) * random.randint(-1,1)
    rand_y = int((random.uniform(0,1.5)/10.0)*height) * random.randint(-1,1)
    shifted_image = Image.new("RGB",(width + rand_x ,height + rand_y))
    shifted_image.paste(img,(rand_x, rand_y))
    shifted_bbox = np.array(bbox) + np.array([rand_x, rand_y] * 2)
    return shifted_image, shifted_bbox.tolist()
    
def resize_image(img, target_size, bbox=None):
    #img = Image.fromarray(img, 'RGB')
    if bbox != None:
        aug_img, aug_bbox = augment_image(img, bbox)
    	new_bbox =np.array((np.array(aug_bbox, dtype = float) / np.array(aug_img.size *2, dtype = float)) * np.array(target_size * 2,dtype = int), dtype = int)
    	return aug_img.resize(target_size), new_bbox
    else: 
    	return img.resize(target_size)

def generator(path, df,target_size, batch_size, shuffle = True):
    count = 0
    df['labels'] = df['Annotation tag'].replace(to_replace = ['stop', 'speedLimit', 'pedestrianCrossing'], value=[0,1,2])
    while True:
        sampled_data = df.sample(n = batch_size, replace = True)
        sampled_data.reset_index(inplace=True)
        bbox, c = [], []
        imgs = []
        for _,i in sampled_data.iterrows():
    #         img_temp = Image.open(path +'train/' + str(i['Annotation tag'])+'/'+str(i['Filename'])) 
            img_temp = Image.open(path + str(i['Annotation tag'])+'/'+str(i['Filename'])) 
            bbox_temp = [i['Upper left corner X'],i['Upper left corner Y'],i['Lower right corner X'],i['Lower right corner Y']]
            img_new_temp, bbox_new_temp = resize_image(img_temp,target_size, bbox_temp) 
            imgs.append(np.array(img_new_temp))
            bbox.append(bbox_new_temp)
            c.append(i['labels'])
        count += batch_size
        yield np.asarray(imgs), [np.asarray(to_categorical(c,3)), np.asarray(bbox)]

def test_images(path, df, target_size, batch_size, shuffle = True):
    df['labels'] = df['Annotation tag'].replace(to_replace = ['stop','speedLimit', 'pedestrianCrossing'],value = [0,1,2])
    test_imgs = []
    yss = []
    for _, i in df.iterrows():
        test_img = Image.open(path + str(i['Annotation tag'])+'/'+str(i['Filename']))
        new_test_img = resize_image(test_img, target_size)
        test_imgs.append(np.array(new_test_img))
        yss.append(i['labels'])
    return np.asarray(test_imgs), np.asarray(to_categorical(yss,3))


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
    
