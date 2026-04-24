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
EPOCHS = 5
INCLUDE_LIST = ['ship', 'truck']
SHOW_GRAPHS = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
##### Functions and Classes #####
#################################

class CIFAR10_NET(nn.Module):
    """
    Define the neural network used for predictions on the CIFAR10 dataset.
    """

    def __init__(self, num_classes):
        super(CIFAR10_NET, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3))
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        # self.mpool = nn.MaxPool2d(kernel_size=2)
        # self.drop1 = nn.Dropout(p=0.25)
        # self.flat = nn.Flatten()
        # self.fc1 = nn.Linear(in_features=64*12*12, out_features=128)
        # self.drop2 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.mpool(x)
        # x = self.drop1(x)
        # x = self.flat(x)
        # x = F.relu(self.fc1(x))
        # x = self.drop2(x)
        # x = self.fc2(x)
        return x
    

def custom_accuracy(testloader, model):
    """
    A custom accuracy function using code from the EEE419 PyTorch examples.
    """

    num_correct = 0
    num_attempts = 0

    for images, labels in testloader:

        with torch.no_grad():

            outputs = model(images)
            guesses = torch.argmax(outputs, 1)

            num_guess = len(guesses)
            num_right = torch.sum(labels == guesses).item()

            num_correct  += num_right
            num_attempts += num_guess

    accuracy = 100*num_correct/num_attempts
    return accuracy

def training_loop(transform):
    """
    Define a training loop which will be used for the RGB and Grayscale situations.
    """

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

    # Create an instance of the CIFAR10_NET class and send it to the GPU if available.
    model = CIFAR10_NET(NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

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
        for batch_idx, (images, labels) in enumerate(trainloader):

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

        print(f'Epoch {epoch}/{EPOCHS}:\n\tTraining Loss = {train_loss}\n\tTesting Loss = {test_loss}')

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

    # Return the model accuracy and training time.
    accuracy = custom_accuracy(testloader, model)
    duration = t_start - t_end
    return accuracy, duration

####################
##### RGB Case #####
####################

# Create the data transform based on tensor conversion and normalization.
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5], [0.5])
transform = transforms.Compose([to_tensor, normalize])

# Run the training loop with this transform, which leaves the RGB channels intact.
rgb_acc, rgb_time = training_loop(transform)

##########################
##### Grayscale Case #####
##########################

# The grayscale code differs only in the transformation. I have added a grayscale transform before normalization.
to_tensor = transforms.ToTensor()
grayscale = transforms.Grayscale(num_output_channels=1)
normalize = transforms.Normalize([0.5], [0.5])
transform = transforms.Compose([to_tensor, grayscale, normalize])

# Run the training loop with this new transform.
gray_acc, gray_time = training_loop(transform)

#######################################################
##### Display the results in the requested format #####
#######################################################

print(f'RGB Accuracy: {rgb_acc:.1f}%')
print(f'Grayscale Accuracy: {gray_acc:.1f}%')
print(f'RGB Runtime: {rgb_time:.5f} seconds')
print(f'Grayscale Runtime: {gray_time:.5f} seconds')
print('Steven Wallace recommends RGB algorithm')