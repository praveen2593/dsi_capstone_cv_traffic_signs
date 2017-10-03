
# Traffic Sign Detection and Classification Using Convolutional Neural Networks

## Overview
Around 1.25 million people die every year due to traffic accidents. Human negligence is found to be one of the biggest reason. Autonomous cars could solve this problem, but one of the challenges they face is detecting and identifying traffic signs. Although many have researched in the same topic, most of them use images of just street signs where there is very less noise and a lot of signal. But in a real world application, cars experience more noise (roads, incoming traffic, trees, buildings etc) than the signal (traffic sign).

Through this project, I wanted to see how Convolutional Neural Network models perform when they come across such images which have very less signal, and identify ways in which I can improve their performance. 

## Dataset
The data was obtained from University of California, San Diego's Laboroatory of Safe and Intelligent Automobiles (LISA). Their traffic sign dataset consisted of 6610 annotated frames of 47 different types of traffic signs in the US. The images were obtained from different cameras, hence their sizes varied from 640x480 to 1024x522 pixels. Each sign had information about the sign type position and size of the image which was available as a separate text file. 

## Pre-Processing
For the final model, I built my own image Data generator function, which performed augmentation, resizing processes and calculated the new bounding box for the image and returns in batches.

## Metrics and Loss Functions
For the loss function to train the model, I used a modified mean squared error for the bounding box head, and softmax for the classification head. For the metrics, since it was a multiclass classification problem, accuracy did not give me accurate results. Keras did not implement precision and recall for the same. Hence I used the loss function, as it had more accurate information than other metrics. For the test data, I computed the individual precision and recall rate for each class.

## Model Development
I initially started of with a transfer learning model, but its performance was pretty low, as there was a lot of noise in the data. Hence I built my own architecture and implemented a version of the Fast-RCNN method, the final model has a bounding box regressor output and a classification output, whose combined loss changes the weights of the model. Hence when predicting, I remove the bounding box regressor head, thereby having only a classification output. This model proved to be the most effective.

## Result and Inference
For my test data, the STOP sign class had a high recall value. But still my model was not able to accurately predict the Speed Limit sign and the Pedestrian Crossing sign. Going through the data I attributed the poor performance to:
1. The speed limit class had different values in the signs (40, 45, 50, 55, 65 etc)
2. The shape, colour and position of the pedestrian crossing signs were such that they blended with the surroundings making it hard to identify.

## Files in src and it's use

* final_model.py - Predicts the fraud risk level based on the probability

## Rough timeline

* Week 1: Worked on Transfer Learning, used VGG to build my base model. Read through multiple research papers to find the optimal way to increase performance. Performed Augmentation processes.
* Week 2: Created my own model, and overcame various roadblocks. Improved performance by implementing the Fast-RCNN approach. Implemented my own augmentation function, to improve performance of model.

