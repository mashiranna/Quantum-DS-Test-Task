# Soil erosion detection
## Code desciption
Fist of all, we need to prepare the data. It was made according to this guide: https://medium.datadriveninvestor.com/preparing-aerial-imagery-for-crop-classification-ce05d3601c68.
Then we need to get some train data and test data. To achive this we can split the given data into several small pieces, using the rasterio library.
After we split that sveral pieces into train data and test data, using scikit-learn.

There is some description about used libraries:
tensorflow is the main module of the TensorFlow library. tensorflow.keras is the Keras API implementation of TensorFlow.
backend module provides low-level operations for building deep learning models.
segmentation_models is a third-party library that provides pre-implemented models for semantic segmentation tasks.

Also we define two hyperparameters for the Focal loss function: 
ALPHA controls the balance between the positive and negative class weights in the loss function. 
GAMMA is a focal factor that adjusts the focus on hard examples during training.

The FocalLoss function takes two inputs: targets and inputs, which represent the true and predicted values, respectively. 
The function calculates the binary crossentropy loss between the targets and inputs using the K.binary_crossentropy function. 
The BCE is then exponentiated using K.exp and multiplied by a term that applies the Focal Loss weighting, which is determined by the alpha and gamma hyperparameters. 
The result is then averaged over all the input values using K.mean and returned as the Focal Loss.

Next, the code defines the dice coefficient function. The dice coefficient is then calculated as the mean of the dice score across all samples.

The code then defines the callbacks list, which contains three callbacks that will be used during the model training process.
The first callback, ModelCheckpoint, will save the best model weights during training based on the validation loss. 
The second callback, ReduceLROnPlateau, will reduce the learning rate of the optimizer when the validation loss stops improving. 
The third callback, EarlyStopping, is commented out in this code, but it can be used to stop the model training early if the validation loss does not improve after a certain number of epochs.

The code then defines two callbacks:
ModelCheckpoint saves the best model weights based on the validation loss.
ReduceLROnPlateau reduces the learning rate of the optimizer when the validation loss plateaus.

Finally, the code defines the model using the Unet function from the segmentation_models library. 
This function creates a UNet model architecture based on the EfficientNet-B0 backbone, with one output class, since it is a binary segmentation task, and a sigmoid activation function. 
 The model is then compiled with the Adam optimizer and the Focal Loss function as the loss function. 
The model also calculates two metrics: 
Mean Intersection over Union (IoU) and Dice Coefficient. 
The fit_generator function is then called to train the model using a data generator, that is created by the make_image_gen function. The epochs parameter specifies the number of training epochs, and the steps_per_epoch parameter 
specifies the number of steps (batches) per epoch. The callbacks parameter specifies the list of callbacks to use during training, 
and the validation_data parameter specifies the validation data to use during training.
## Results
Detection after 10 epoches:

![image](https://user-images.githubusercontent.com/55777589/226716408-b51481bc-fb14-4e5e-8b96-396d4b639c7d.png)
![image](https://user-images.githubusercontent.com/55777589/226716524-0e78d4a6-207c-48ad-b92e-d2e839c18d96.png)
![image](https://user-images.githubusercontent.com/55777589/226716551-445ddb1d-82d3-462e-a1a3-ea0a26b0e929.png)
![image](https://user-images.githubusercontent.com/55777589/226716572-60a2d88a-39d0-4b20-8828-fcf3fb55c957.png)
![image](https://user-images.githubusercontent.com/55777589/226716586-786d9e11-5f43-482e-bd2c-6788af450cd6.png)

At this moment, we can see the model functions correctly, but definetly needs more epoches and steps per epoch to achive more accuracy.  
## Requirements 
The library requirements are descripted in requirements.txt. The T36UXV_20200406T083559_TCI_10m.jp2 file, masks and Soil erosion detection.ipynb have to be placed in one folder. Also, you need to create two folders named 'train' and 'mask_train' to place there splited data. 
## Soloution report

Soil erosion is a significant environmental challenge that can lead to significant soil degradation and the loss of important ecosystem services. Remote sensing data and machine learning techniques have been increasingly used to detect soil erosion and to monitor soil erosion processes.
Here are some proposals and results from different papers about the soil erosion detection problem:

- Use of high-resolution satellite imagery;
- Fusion of multi-source data: Combining data from different sources, such as satellite imagery and ground-based measurements, can improve the accuracy of soil erosion detection;
- Use of deep learning techniques: Deep learning techniques, such as convolutional neural networks, have been used to detect soil erosion from remote sensing data;

In conclusion, the combination of high-resolution satellite imagery, machine learning techniques, multi-source data fusion, and deep learning techniques can help to detect soil erosion and to monitor soil erosion processes effectively. However, further research is needed to develop more accurate and robust soil erosion detection methods that can be applied in different regions and under different conditions.
