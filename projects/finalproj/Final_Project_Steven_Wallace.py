# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 23 April 2026

# Final Project

# I did not use AI at all to complete this assignment.

# Another example of how to train a neural network on CIFAR10.
# https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py

# How to extract a subset of the CIFAR10 dataset.
# https://stackoverflow.com/questions/54380140/how-do-i-extract-only-subset-of-classes-from-torchvision-datasets-cifar10#54380927

# Build your first pytorch model in minutes.
# https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier

import torch
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F
import timm

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

#####################
##### Constants #####
#####################

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 25

INCLUDE_LIST = ['ship', 'truck']
SHOW_GRAPHS = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
##### Functions and Classes #####
#################################

class CIFAR10_RGB_Net(nn.Module):
    """
    Define the neural network used for predictions on the RGB dataset. This inherits from nn.Module and initializes a pretrained model from timm called 'efficientnet_b0.'

    Attributes
    ----------
    __init__: instance method
        Initialize the pretrained neural network based upon efficientnet_b0. I am removing the final layer here and replacing it with a linear classifier.
    forward: instance method
        Determine the output predictions based on an input to the neural network "x". This is used with back propagation to form the training loop.
    """

    def __init__(self, num_classes):
        super(CIFAR10_RGB_Net, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class CIFAR10_Gray_Net(nn.Module):
    """
    Define the neural network used for predictions on the Grayscale dataset. This is a modified version of the EEE419 example code.

    Attributes
    ----------
    __init__: instance method
        Define the architecture of the neural network, which involves convolutional 2D layers, pooling, dropout, and flattening.
    forward: instance method
        Define what it means to forward propagate this neural network. It makes use of RELU as the activation function.

    """

    def __init__(self, num_classes: int):
        super(CIFAR10_Gray_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.mpool = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.25)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*14*14, out_features=128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mpool(x)
        x = self.drop1(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x
    
def custom_accuracy(testloader, model):
    """
    A custom accuracy function using code from the EEE419 PyTorch examples.

    Parameters
    ----------
    testloader: torch.utils.data.DataLoader
        The testing dataset loading object, which groups the testing data into batches so the model can be tested.
    model: nn.Module
        The neural network that is being evaluated in this situation, which is either the RGB or the grayscale variant.

    Returns
    -------
    accuracy: float
        The accuracy of the model based upon these testing data.
    """

    # Initialize variables to keep track of the number of attempts and correct answers.
    num_correct = 0
    num_attempts = 0

    # Set the model to the evaluate mode. It is important we do not touch the weights or biases right now.
    model.eval()
    with torch.no_grad():

        # Loop over the batches defined by the testloader.
        for images, labels in testloader:

            # Send the images and labels to the GPU if available.
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Count the number of times we guess the result correctly, then calculate an accuracy.
            outputs = model(images)
            guesses = torch.argmax(outputs, 1)
            num_guess = len(guesses)
            num_right = torch.sum(labels == guesses).item()
            num_correct  += num_right
            num_attempts += num_guess

    # Calculate and return the accuracy, which is defined by the number of correct answers divided by the number of attempts.
    accuracy = 100*num_correct / num_attempts
    return accuracy

def training_loop(model, trainloader, testloader, criterion, optimizer):
    """
    Define a training loop which will be used for the RGB and Grayscale situations.

    Parameters
    ----------
    model: nn.Module
        The neural network that is being evaluated in this situation, which is either the RGB or the grayscale variant.
    trainloader: torch.utils.data.DataLoader
        The training dataset loading object, which groups the training data into batches so the model can be trained.
    testloader: torch.utils.data.DataLoader
        The testing dataset loading object, which groups the testing data into batches so the model can be tested.
    criterion: nn.CrossEntropyLoss
        The criterion we are using to evaluate the model, also called the loss function. We are using CrossEntropyLoss in this case.
    optimizer: optim.Adadelta
        The technique we are using to optimize the model during back-propagation, and we are using Adadelta in this program.

    Returns
    -------
    accuracy: float
        The accuracy of the model based upon these testing data.
    duration: float
        The length of time it took to train the model and evaluate its performance.
    """

    # Initialize lists to store the training and testing losses, so they can be graphed later.
    train_losses = list()
    test_losses = list()

    # Record the time when training began.
    t_start = time.time()

    # Loop over the entire dataset "EPOCHS" times.
    for epoch in range(EPOCHS):

        # Train the model for this epoch.
        model.train()
        running_loss = 0.0

        # Begin to loop over the training loader.
        for images, labels in trainloader:

            # Send the images and labels to the GPU if available.
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Carry out forward and back propagation on the neural network.
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        # Keep record of the training loss for each epoch.
        train_loss = running_loss / len(trainloader.dataset)
        train_losses.append(train_loss)

        # Evaluate the model for this epoch.
        model.eval()
        running_loss = 0.0

        # Begin to loop over the testing loader, making sure to leave the weights and biases untouched.
        with torch.no_grad():
            for images, labels in testloader:

                # Move the inputs and images to the GPU if available.
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        # Keep track of the testing loss for each epoch so that it can be graphed later.
        test_loss = running_loss / len(trainloader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{EPOCHS}:\n\tTraining Loss = {train_loss}\n\tTesting Loss = {test_loss}')

    # Record the time when training ended.
    t_end = time.time()

    # Plot the training and test losses vs epoch number. This gives me information about overfitting or underfitting.
    if SHOW_GRAPHS:
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Testing Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Find the accuracy and training duration, then return those values.
    accuracy = custom_accuracy(testloader, model)
    duration = t_end - t_start
    return accuracy, duration

####################
##### RGB Case #####
####################

# Create the data transform based on tensor conversion and normalization.
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5], [0.5])
transform = transforms.Compose([to_tensor, normalize])

# Load the training data and apply the requested transform.
trainset = torchvision.datasets.CIFAR10(root='~/CIFAR10_data', train=True, 
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                        shuffle=True)

# Load the testing data and apply the requested transform.
testset = torchvision.datasets.CIFAR10(root='~/CIFAR10_data', train=False, 
                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False)

# Create an instance of the CIFAR10_RGB_Net class and send it to the GPU if available.
model = CIFAR10_RGB_Net(NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

# Run the training loop with this model and transform, leaving the RGB channels intact.
rgb_acc, rgb_time = training_loop(model, trainloader, testloader, criterion, optimizer)

##########################
##### Grayscale Case #####
##########################

# The grayscale code differs only in the transformation. I have added a grayscale transform before normalization.
to_tensor = transforms.ToTensor()
grayscale = transforms.Grayscale(num_output_channels=1)
normalize = transforms.Normalize([0.5], [0.5])
transform = transforms.Compose([to_tensor, grayscale, normalize])

# Load the training data and apply the requested transform.
trainset = torchvision.datasets.CIFAR10(root='~/CIFAR10_data', train=True, 
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                        shuffle=True)

# Load the testing data and apply the requested transform.
testset = torchvision.datasets.CIFAR10(root='~/CIFAR10_data', train=False, 
                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False)

# Create an instance of the CIFAR10_RGB_Net class and send it to the GPU if available.
model = CIFAR10_Gray_Net(NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

# Run the training loop with this new model and transformed data.
gray_acc, gray_time = training_loop(model, trainloader, testloader, criterion, optimizer)

#######################################################
##### Display the results in the requested format #####
#######################################################

print(f'RGB Accuracy: {rgb_acc:.1f}%')
print(f'Grayscale Accuracy: {gray_acc:.1f}%')
print(f'RGB Runtime: {rgb_time:.5f} seconds')
print(f'Grayscale Runtime: {gray_time:.5f} seconds')
print('Steven Wallace recommends RGB algorithm')