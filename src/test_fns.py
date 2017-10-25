### This file contains functions used for test data 

def testing_result(df, model):
    '''Calculates test set results'''
    test_fn = image_names('/home/ubuntu/test/')
    test_df = data_splitter(test_fn, df)
    test_imgs, test_labels = test_images('/home/ubuntu/test/', test_df, (IMG_SIZE, IMG_SIZE), 8)
    test_pred = model.predict(x = test_imgs)
    test_pred_cat = convert_tocat(test_pred)
    #Calculating Individual Precision
    a = np.array(test_pred_cat).T
    b = np.array(test_labels).T
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


def test_images(path, df, target_size, batch_size, shuffle = True):
	'''Generator Function for test images'''
    df['labels'] = df['Annotation tag'].replace(to_replace = ['stop','speedLimit', 'pedestrianCrossing'],value = [0,1,2])
    test_imgs = []
    yss = []
    for _, i in df.iterrows():
        test_img = Image.open(path + str(i['Annotation tag'])+'/'+str(i['Filename']))
        new_test_img = resize_image(test_img, target_size)
        test_imgs.append(np.array(new_test_img))
        yss.append(i['labels'])
    return np.asarray(test_imgs), np.asarray(to_categorical(yss,3))


