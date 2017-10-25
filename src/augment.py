### Functions to augment images and create generators

def augment_image(img, bbox):
    ''' Custom function which augments image and calculates bounding box information for the augmented image'''
    width, height = img.size
    rand_x = int((random.uniform(0,1.5)/10.0)*width) * random.randint(-1,1)
    rand_y = int((random.uniform(0,1.5)/10.0)*height) * random.randint(-1,1)
    shifted_image = Image.new("RGB",(width + rand_x ,height + rand_y))
    shifted_image.paste(img,(rand_x, rand_y))
    shifted_bbox = np.array(bbox) + np.array([rand_x, rand_y] * 2)
    return shifted_image, shifted_bbox.tolist()


def resize_image(img, target_size, bbox=None):
    '''Calculates new bounding box information based on augmentation'''
    if bbox != None:
        aug_img, aug_bbox = augment_image(img, bbox)
        new_bbox =np.array((np.array(aug_bbox, dtype = float) / np.array(aug_img.size *2, dtype = float)) * np.array(target_size * 2,dtype = int), dtype = int)
        return aug_img.resize(target_size), new_bbox
    else:
        return img.resize(target_size)


def generator(path, df,target_size, batch_size, shuffle = True):
    ''' Custom Generator function which provides batches of data'''
    count = 0
    df['labels'] = df['Annotation tag'].replace(to_replace = ['stop', 'speedLimit', 'pedestrianCrossing'], value=[0,1,2])
    while True:
        sampled_data = df.sample(n = batch_size, replace = True)
        sampled_data.reset_index(inplace=True)
        bbox, c = [], []
        imgs = []
        for _,i in sampled_data.iterrows():
            img_temp = Image.open(path + str(i['Annotation tag'])+'/'+str(i['Filename']))
            bbox_temp = [i['Upper left corner X'],i['Upper left corner Y'],i['Lower right corner X'],i['Lower right corner Y']]
            img_new_temp, bbox_new_temp = resize_image(img_temp,target_size, bbox_temp)
            imgs.append(np.array(img_new_temp))
            bbox.append(bbox_new_temp)
            c.append(i['labels'])
        count += batch_size
        yield np.asarray(imgs), [np.asarray(to_categorical(c,3)), np.asarray(bbox)]
