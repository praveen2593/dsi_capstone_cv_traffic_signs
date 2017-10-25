### Contains utility functions used in final model

def convert_tocat(test_pred):
	''' Converts input to matrix with 3 columns to match the format of the test_labels'''
    test_pred_cat = []
    for i in test_pred[0]:
        if np.argmax(i) == 0:
            test_pred_cat.append([1,0,0])
        elif np.argmax(i) == 1:
            test_pred_cat.append([0,1,0])
        else:
            test_pred_cat.append([0,0,1])
    return test_pred_cat


def msetf(y_true, y_pred):
    '''Loss Function for Bounding Box'''
    return mean_squared_error(y_true, y_pred)/25000.0


def get_data():
    ''' Getting and Cleaning data'''
    df = pd.read_csv('allAnnotations.csv', sep=';')
    df.drop_duplicates(subset='Filename', inplace = True)
    df = df.loc[df['Annotation tag'].isin(['stop', 'pedestrianCrossing', 'speedLimitUrdbl','speedLimit25','speedLimit35','speedLimit45','speedLimit15','speedLimit40','speedLimit50',            'speedLimit55','speedLimit30','speedLimit65'])]
    df['Filename'] = [i.split('/')[2] for i in df['Filename']]
    df.drop(['Occluded,On another road', 'Origin frame number', 'Origin track','Origin track frame number','Origin file'], axis = 1, inplace = True)
    df['Annotation tag'] = [i.split('_')[0] for i in df['Filename']]
    df['Annotation tag'].replace(to_replace = ['pedestrian'], value = 'pedestrianCrossing',inplace = True)
    df['Annotation tag'].replace(to_replace = ['speedLimitUrdbl','speedLimit25','speedLimit35','speedLimit45','speedLimit15','speedLimit40','speedLimit50','speedLimit55','speedLimit30',        'speedLimit65'], value='speedLimit', inplace= True)
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
    for classes in listdir(path)[:3]:
        filename.append(listdir(path+str(classes)+'/'))
    return filename


def data_splitter(list_names,df):
    '''Creating new dataframe based on values in a particular column'''
    new_df = df.loc[df['Filename'].isin(list_names[0]) | df['Filename'].isin(list_names[1]) | df['Filename'].isin(list_names[2])]
    new_df.reset_index(inplace=True)
    return new_df



