from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras import backend as K
#K.set_image_data_format('channels_first')
from glob import glob
import numpy as np
from keras.preprocessing import image

#Parameters to tune
IMG_SIZE = 350
batch_size = 32
epochs = 30
NUM_CLASSES = 3
NUM_IMAGES = 10000

def get_batches(path, gen=image.ImageDataGenerator(width_shift_range=0.3,height_shift_range=0.3), shuffle=True, batch_size=8, class_mode='categorical'):
    """
        Takes the path to a directory, and generates batches of augmented/normalized data. 
	Yields batches indefinitely, in an infinite loop.

        See Keras documentation: https://keras.io/preprocessing/image/
    """
    return gen.flow_from_directory(path, target_size=(IMG_SIZE,IMG_SIZE), class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def cnn_model():
    model = Sequential()
    #Block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #Block2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    #Block 3
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    #Block 4
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    #FC Block
    model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def fit(model, batches, val_batches, nb_epoch=epochs):
    """
        Fits the model on data yielded batch-by-batch by a Python generator.
        See Keras documentation: https://keras.io/models/model/
    """
    return model.fit_generator(batches, steps_per_epoch=NUM_IMAGES/epochs,
            validation_data=val_batches, validation_steps = 550/30, epochs=nb_epoch)

def create():
    """
	Creates CNN Model and compiles with optimizer
    """
    model = cnn_model()
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def get_data():
    '''
	Gets data and converts to batch generators
    '''
    path =''
    batches = get_batches('train', batch_size=batch_size)
    val_batches = get_batches(path+'val', batch_size=batch_size)
    test_batches = get_batches(path+'test', batch_size = batch_size)
    return batches, val_batches, test_batches

if __name__ == '__main__':
    model = create()
    train_generator, validation_generator, test_generator = get_data()
    fit_model = fit(model, train_generator, validation_generator, test_generator, nb_epoch = 1)
