# example mnist implementation using PyTorch
# author hbm updated by sdm

import torch                         # import the various PyTorch packages
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from torchvision import datasets     # the data repository
from torchvision import transforms   # tranforming the data

################################################
# Setting up constants for training
################################################
BATCH_SIZE  = 128   # number of samples per gradient update
NUM_CLASSES = 10    # how many classes to classify (10 digits, 0-9)
EPOCHS      = 2     # how many epochs to run trying to improve

################################################
# Create the network
################################################
class MNIST_NET( nn.Module ):

    ################################################
    # Initializing the network
    ################################################
    def __init__( self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.mpool = nn.MaxPool2d( kernel_size=2 )
        self.drop1 = nn.Dropout( p=0.25 )
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear( in_features=64 * 12 * 12, out_features=128 )
        self.drop2 = nn.Dropout( p=0.5 )
        self.fc2   = nn.Linear( in_features=128, out_features=num_classes )

    ################################################
    # Forward pass of the network
    ################################################
    def forward( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = self.mpool( x )
        x = self.drop1( x )
        x = self.flat(  x )
        x = F.relu( self.fc1( x ) )
        x = self.drop2( x )
        x = self.fc2(   x )
        return x

###################################################
# steps required to transform the data:
# 1. Convert to a tensor
# 2. Transform with mean and standard deviation 0.5
# 3. Bundle the steps together
###################################################

to_tensor   = transforms.ToTensor()
normalize   = transforms.Normalize([0.5],[0.5])
transform   = transforms.Compose( [ to_tensor, normalize ] )

###################################################
# load the training data and transform it
# then get it into the pytorch environment
# then do the same for the test data
###################################################

trainset    = datasets.MNIST('~/MNIST_data/train', download=True,
                             train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)
testset     = datasets.MNIST('~/MNIST_data/test', download=True,
                             train=False, transform=transform)
testloader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=True)

################################################
# Get ready to train
################################################

mnist_net = MNIST_NET( NUM_CLASSES )                  # initialize the object
criterion = nn.CrossEntropyLoss()                     # select cost function
optimizer = optim.Adadelta( mnist_net.parameters() )  # select the optimizer

################################################
# Training loop
################################################

num_batches = len(trainloader)

for epoch in range( EPOCHS ):

    for batch_idx, (images, labels) in enumerate(trainloader):

        optimizer.zero_grad()                     # resets gradient optimizer
        output = mnist_net( images )              # calls the forward function
        loss   = criterion( output, labels )      # calculates the errors
        loss.backward()                           # back propagates the changes
        optimizer.step()                          # updates the weights

        if batch_idx % 100 == 0:                  # report periodically
            batch_loss = loss.mean().item()
            print("Epoch {}/{}\tBatch {}/{}\tLoss: {}" \
                  .format(epoch, EPOCHS, batch_idx, num_batches, batch_loss))

################################################
# Testing loop
################################################

num_correct  = 0   # initialize the counters
num_attempts = 0

for images, labels in testloader:  # for each batch of images

    with torch.no_grad():

        outputs   = mnist_net( images )         # run the images through
        guesses   = torch.argmax( outputs, 1)   # most probable guess per image

        num_guess = len( guesses )              # how many in this batch
        num_right = torch.sum( labels == guesses ).item()  # how many correct

        num_correct  += num_right               # track accuracy
        num_attempts += num_guess

print("Total test accuracy:", 100*num_correct/num_attempts,"%")
