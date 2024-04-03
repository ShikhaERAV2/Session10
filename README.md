## Custom Resnet Model- CIFAR-10 Image Classification using PyTorch
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into 391 training batches and 71 test batch, each with 128 images. The test batch contains exactly randomly-selected images from each class. The training batches contain the remaining images in random order. 

The various classes are ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

Here are the classes in the dataset, as well as 10 random images from each: 
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/92783e22-2d42-4bfd-8c31-b26040152108)

## Model Description
This is a Multiple convolution layers in Convolutional Neural Network for Image identification trained on CIFAR10 dataset.Basic model structure 

1. PrepLayer -

    Conv 3x3 s1, p1) >> BN >> RELU [64k]
3. Layer1 -

    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]

    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 

    Add(X, R1)
4. Layer 2 -

    Conv 3x3 [256k] >> MaxPooling2D >> BN >> ReLU
6. Layer 3 -

    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]

    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]

    Add(X, R2)
7. MaxPooling with Kernel Size 4
8. FC Layer 
9. SoftMax

## Code Structure
- S10_V6.ipynb: The main Jupyter Notebook contains the code to load data in train and test datasets -> transform data-> load model (defined in model.py)-> train model -> test the model -> Check the accuracy of the model thus trained. This model is a custom Resnet implementation. 
- model.py: This file contains the definition of the model. Basic architecture of the model is defined with multiple convolution layers and fully connected layers.
- utils.py: This file contains the utility functions like display of the sample data images and plotting the accuracy and loss during training.

## Requirements
 - Pytorch
 - Matplotlib

## Model
Model Name : ResNet_Custom

*Test Accuracy = 90.56% (max)

*Train Accuracy = 96.55%

*Total params: 6,573,130

### Analysis:

- Model is Overfitting.
- Number of parameter is high though model is reaching the 90% accuracy in desired epoch .
- Batch Normailzation with image augmentation helps improve the model performance.
- ADAM optimizer improved the accuracy with provided hyperparameters.

## LR Finder:

![image](https://github.com/ShikhaERAV2/Session10/assets/160948226/782417b9-e672-4efb-b4be-5820252b1d6e)


## Model Performance:

![image](https://github.com/ShikhaERAV2/Session10/assets/160948226/4c4dad18-80ee-4dc1-ae02-24fe695de2bf)



## Mis-Classified Images:

![image](https://github.com/ShikhaERAV2/Session10/assets/160948226/b5ea079f-4f2c-489e-b5e6-5cba057dc9a7)
