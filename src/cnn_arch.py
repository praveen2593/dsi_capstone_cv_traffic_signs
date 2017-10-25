### Function to create Architecture and Compile model

def create_model():
    '''Creating Architecture of the model'''
    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 3))
    #Block 1
    layer_1 = Conv2D(32,(3,3), padding = 'same', activation = 'relu')(inputs)
    layer_2 = Conv2D(32,(3,3), padding = 'same', activation = 'relu')(layer_1)
    layer_3 = MaxPooling2D(pool_size = (2,2))(layer_2)
    #Block 2
    layer_4 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(layer_3)
    layer_5 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(layer_4)
    layer_6 = MaxPooling2D(pool_size = (2,2))(layer_5)
    #Block 3
    layer_7 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(layer_6)
    layer_8 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(layer_7)
    layer_9 = MaxPooling2D(pool_size = (2,2))(layer_8)
    #Final Block
    layer_19 = Flatten()(layer_9)
    layer_20 = Dense(512, activation = 'relu')(layer_19)
    softmax_class = Dense(NUM_CLASSES, activation = 'softmax', name= 'softmax_class')(layer_20)
    relu_bbox = Dense(4, activation = 'relu', name = 'relu_bbox')(layer_20)
    #Creating model with 2 inputs and above architecture
    model = Model(inputs= inputs, outputs = [softmax_class, relu_bbox])
    return model


def compile_model():
    '''Compiling model'''
    model = create_model()
    adam = Adam(lr = 1e-8, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay=0.01)
    model.compile(loss={'relu_bbox':msetf, 'softmax_class':'categorical_crossentropy'}, optimizer=adam, metrics = ['categorical_accuracy'])
    return model
