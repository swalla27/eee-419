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

import torch
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import sys
import os

#####################
##### Constants #####
#####################

BATCH_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 2
INCLUDE_LIST = ['ship', 'truck']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

###################
##### Classes #####
###################

class CIFAR10_NET(nn.Module):
    """
    Define the neural network.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.mpool = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(p=0.25 )
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=128)
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

###########################
##### Gather the data #####
###########################

# Create the data transform based on tensor conversion and normalization.
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5], [0.5])
transform = transforms.Compose([to_tensor, normalize])

# Load the training data.
trainset = torchvision.datasets.CIFAR10(root='~/CIFAR10_data', train=True, 
                            download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=False)


# for batch_idx, (images, labels) in enumerate(trainloader):

#     plt.imshow(images[0])
#     plt.title('original photo')
#     plt.show()

#     print(labels)
#     print(batch_idx)

# print(dir(trainloader))

# sys.exit()

# # Remove the unwanted classes from the dataset.
# labels = np.array(trainset.classes)
# include = np.array(INCLUDE_LIST).reshape(1, -1)
# mask = (labels.reshape(-1, 1) == include).any(axis=1)
# # trainset.data = trainset.data[mask]
# # trainset.classes = trainset.classes[mask]

# print(labels)
# print(trainset.classes)
# print(trainset.meta)
# print(trainset.data.shape)
# print(include)
# print(mask)

# sys.exit()



# Load the testing data.
testset = torchvision.datasets.CIFAR10(root='~/CIFAR10_data', train=False, 
                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=True)

#####################################
##### Create the neural network #####
#####################################



