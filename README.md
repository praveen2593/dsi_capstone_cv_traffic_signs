###### Project Author: Praveen
###### Date: 8/15/2017

## Traffic Signal Detection and Classification


#### Description

Through this project I am interested in identifying traffic signs in a given image, and classify them based on their type.

My motivation to work on this problem:
- Understanding Neural Networks and Deep Learning, while learning their necessity over other machine learning models
- Intersection of my interest in the automotive industry and machine learning
- Solving complex problems which are currently worked on in the industry

#### Potential Workflow
My goals for this project are:
- Identify and classify traffic signs present in the images through SVM and Gradient boosting models
- Compare the performance of my model against a pre-trained Convolutional Neural Network model
- (Expecting the pre-trained CNN model to perform better) Modify the CNN model to improve chosen metric
- Build a CNN architecture from scratch, and compare the performance with previous models
- Extend the best model from above to work on videos instead of images for classification

#### Presentation
Visual demonstration of my model by testing with pictures/videos of streets near Galvanize and a presentation on the process

#### Next Step:
Sample my data, and create a SVM model and Gradient Boosting Classifier model.

#### Data:
The dataset and description is provided here:
http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

###### Description of Dataset:
1. 47 US sign types
2. 7855 annotations on 6610 frames.
3. Sign sizes from 6x6 to 167x168 pixels.
4. Images obtained from different cameras. Image sizes vary from 640x480 to 1024x522 pixels.
5. Some images in color and some in grayscale.
6. Full version of the dataset includes videos for all annotated signs.
7. Each sign is annotated with sign type, position, size, occluded (yes/no), on side road (yes/no).
8. All annotations are save in plain text .csv-files.
9. Includes a set of Python tools to handle the annotations and easily extract relevant signs from the dataset.


<p align='center'>
<img src="/img/signalAhead_1330547327.avi_image4.png" alt="Drawing" style="width: 500px;", align:center/>
</p>


###### Data Pipeline:
- Split data into train, test and validation sets
- Perform Data Augmentation if necessary
- Convert images into matrix form
- Pass the matrix through a classification model
- Understand the performance of model and tune for maximum performance
- Perform the above two steps for all the models prescribed in the description section

###### Potential Problems:
From my understanding, all potential problems reported below needs to be worked on based on the performance of the model
- Traffic signs being distributed in the images, and presence of other objects might reduce the accuracy of the model
 - Potential solution: Run a object detection model first, and perform classification on the results from that
- Presence of some signs in only one side of the picture, might train the model to classify that sign only if it's in that direction
 - Potential Solution: Augment available data to include other possibilities
- Data available might not be sufficient to train a CNN from scratch
 - Potential Solution: Perform Data Augmentation by translating or flipping the model. If that does not prove satisfactory, use the German traffic Sign data.
- Dataset might be imbalanced
 - Data Augmentation would also solve this scenario

###### How far do you anticipate to take the project in the allotted time frame?
Within the deadline, I hope to complete building a CNN architecture from scratch and compare its performance against a pre-trained CNN model
