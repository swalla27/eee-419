# ###  Linear Regression:
# import matplotlib.pyplot as plt                       # for plotting
# import numpy as np
# import pandas as pd                                   # for data frame
# from sklearn.model_selection import train_test_split  # split the data
# from sklearn.linear_model import LinearRegression     # algorithm to use
# from sklearn.metrics import mean_squared_error        # data analysis

# X = np.arange(1,10,0.1)
# y = 2*X + np.random.normal(0, 0.0001, len(X))  # Mean=0, Std=1

# X_train, X_test, y_train, y_test = \
#          train_test_split(X,y,test_size=0.3,random_state=0)

# # X must be a col vector, reshape is needed when we have 1 feature:
# X_train = X_train.reshape(-1, 1)
# X_test = X_test.reshape(-1,1)


# plt.scatter(X,y)
# # plt.show()
# # NOTE: LinearRegression works WITHOUT requiring standarization!
# slr = LinearRegression()              # instantiate a linear regression tool
# slr.fit(X_train,y_train)              # fit the data
# y_train_pred = slr.predict(X_train)   # predict the training values
# x_val = np.array([[2]])
# print(x_val)
# y_val = slr.predict(x_val)
# print(f"X={x_val}, y={y_val}")

# y_test_pred = slr.predict(X_test)     # predict the test values
# print('MSE train: %.3f, test: %.3f' % (
#     mean_squared_error(y_train,y_train_pred),
#     mean_squared_error(y_test,y_test_pred)))




# ############ Unsupervised learning:
# ############ Clustering:
# # Example program for clustering analysis using KMeans which tries for
# # groups of equal variance, minimizing sum of squares or distance from a center
# # author: d updated by sdm
# import numpy as np
# from sklearn.datasets import make_blobs        # create clusters of data
# from sklearn.cluster import KMeans             # cluster analysis algorithm
# import matplotlib.pyplot as plt                # so we can plot the data

# ### First - Dataset generation:
# # create 3 blobs (centers) using 2 features (n_features) and 150 samples.
# # cluster_std is the standard deviation of each blob
# # shuffle the samples in random order (rather than 0s, 1s, then 2s)

# X,y = make_blobs(n_samples=150,n_features=2,centers=3,
#                  cluster_std=0.5,shuffle=True,random_state=0)
# print("Dataset = \n", np.column_stack((X, y)))                    # debug print to show samples

# plt.figure(figsize=(12, 5))

# # First plot (different colors)
# plt.subplot(1, 2, 1)
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0', marker='o')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1', marker='o')
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color='green', label='Class 2', marker='o')

# plt.subplot(1,2,2)
# plt.scatter(X[:,0],X[:,1],c='red',marker='o',s=50)     # plot the data
# plt.grid()
# plt.title('kmeans cluster data')
# plt.show()

# ### Second - Clustering:
# # now train using KMeans
# # We will do this 5 times, specifying 1-5 clusters
# # use smart method for initial cluster centers (k-means++)
# # n_init is the number of times to run with different centroid seeds
# # max_iter limits the number of times the algorithm will be run
# # tol is the convergence limit
# # Create the KMeans widget and then fit and predict

# mkrs = ['s','o','v','^','x']   # markers to use for each cluster
# clrs = ['orange','green','blue','purple','gold']
# inertia = []                   # track the Sum of Squares Error (SSE)
# for numcs in range(3,4):
#     km = KMeans(n_clusters=numcs,init='k-means++',
#                 n_init=10,max_iter=300,tol=1e-4,random_state=0)
#     y_km = km.fit_predict(X)
#     inertia.append(km.inertia_)   # built-in measure of SSE

#     for clustnum in range(numcs):
#         # X[y_km==clustnum,0] says use the entry in X if the corresponding value
#         # in y_km is equal to clustnum. Same for the x and y coordinates
#         plt.scatter(X[y_km==clustnum,0],X[y_km==clustnum,1], # select samples
#                     c=clrs[clustnum],                        # pick color
#                     s=50,                                    # marker size
#                     marker=mkrs[clustnum],                   # which marker
#                     label='cluster'+str(clustnum+1))         # which cluster

#     # plot the centers
#     plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
#                 s=250,c='red',marker='*',label='centroids')

#     plt.legend()
#     plt.grid()
#     plt.title('kmeans with ' + str(numcs) + ' clusters')
#     plt.show()

# plt.plot(list(range(1,len(inertia)+1)),inertia,marker='x')
# plt.xlabel('number of clusters')
# plt.ylabel('inertia')
# plt.title('kmeans cluster analysis')
# plt.show()

# #print(X)
# #print(y_km)






# # example mnist implementation using PyTorch
# # author hbm updated by sdm

# import torch                         # import the various PyTorch packages
# import torch.nn            as nn
# import torch.optim         as optim
# import torch.nn.functional as F
# import random

# from torchvision import datasets     # the data repository
# from torchvision import transforms   # tranforming the data

# ################################################
# # Setting up constants for training
# ################################################
# # BATCH_SIZE  = 1   # number of samples per gradient update
# BATCH_SIZE = 8      # ====> Accuracy 96.89% (EPOCHS = 1)
# ### BATCH_SIZE = 8 ====> Accuracy 97.26% (EPOCHS = 2)
# NUM_CLASSES = 10    # how many classes to classify (10 digits, 0-9)
# EPOCHS      = 1     # how many epochs to run trying to improve
# ### EPOCHS = 1 ====> Accuracy 97% (BATCH_SIZE = 128)
# ### EPOCHS = 2 ====> Accuracy 98% (BATCH_SIZE = 128)
# ### EPOCHS = 3 ====> Accuracy 97.5% (BATCH_SIZE = 128)
# ### EPOCHS =  ====> Accuracy 98.42% (BATCH_SIZE = 128)

# ################################################
# # Create the network
# ################################################
# class MNIST_NET( nn.Module ):

#     ################################################
#     # Initializing the network
#     ################################################
#     def __init__( self, num_classes):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
#         self.mpool = nn.MaxPool2d( kernel_size=2 )
#         self.drop1 = nn.Dropout( p=0.25 )
#         self.flat  = nn.Flatten()
#         self.fc1   = nn.Linear( in_features=64 * 12 * 12, out_features=128 )
#         self.drop2 = nn.Dropout( p=0.5 )
#         self.fc2   = nn.Linear( in_features=128, out_features=num_classes )

#     ################################################
#     # Forward pass of the network
#     ################################################
#     def forward( self, x ):
#         x = F.relu( self.conv1( x ) )
#         x = F.relu( self.conv2( x ) )
#         x = self.mpool( x )
#         x = self.drop1( x )
#         x = self.flat(  x )
#         x = F.relu( self.fc1( x ) )
#         x = self.drop2( x )
#         x = self.fc2(   x )
#         return x

# ###################################################
# # steps required to transform the data:
# # 1. Convert to a tensor
# # 2. Transform with mean and standard deviation 0.5
# # 3. Bundle the steps together
# ###################################################

# to_tensor   = transforms.ToTensor()
# normalize   = transforms.Normalize([0.5],[0.5])
# transform   = transforms.Compose( [ to_tensor, normalize ] )

# ###################################################
# # load the training data and transform it
# # then get it into the pytorch environment
# # then do the same for the test data
# ###################################################

# trainset    = datasets.MNIST('~/MNIST_data/train', download=True,
#                              train=True, transform=transform)
# # subset_indices = list(range(1000))
# # # subset_indices = random.sample(range(len(trainset)), 1000)  # Choose the 1000 samples at random
# # trainset = torch.utils.data.Subset(trainset, subset_indices)
# print("Length of Trainset =", len(trainset))
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                           shuffle=True)
# print('Training size =',len(trainloader))
# testset     = datasets.MNIST('~/MNIST_data/test', download=True,
#                              train=False, transform=transform)
# # subset_indices = list(range(100))
# # # subset_indices = random.sample(range(len(testset)), 100)  # Choose the first 100 samples at random
# # testset = torch.utils.data.Subset(testset, subset_indices)
# print(len(testset))
# testloader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                           shuffle=True)
# print('Testing size =',len(testloader))

# ################################################
# # Get ready to train
# ################################################

# mnist_net = MNIST_NET( NUM_CLASSES )                  # initialize the object
# criterion = nn.CrossEntropyLoss()                     # select cost function
# optimizer = optim.Adadelta( mnist_net.parameters() )  # select the optimizer

# ################################################
# # Training loop
# ################################################

# num_batches = len(trainloader)

# for epoch in range( EPOCHS ):

#     for batch_idx, (images, labels) in enumerate(trainloader):
#         # print(images)
#         # print(labels)

#         optimizer.zero_grad()                     # resets gradient optimizer
#         output = mnist_net( images )              # calls the forward function
#         loss   = criterion( output, labels )      # calculates the errors
#         loss.backward()                           # back propagates the changes
#         optimizer.step()                          # updates the weights

#         if batch_idx % 100 == 0:                  # report periodically
#             batch_loss = loss.mean().item()
#             print("Epoch {}/{}\tBatch {}/{}\tLoss: {}" \
#                   .format(epoch, EPOCHS, batch_idx, num_batches, batch_loss))

# ################################################
# # Testing loop
# ################################################

# num_correct  = 0   # initialize the counters
# num_attempts = 0

# for images, labels in testloader:  # for each batch of images

#     with torch.no_grad():

#         outputs   = mnist_net( images )         # run the images through
#         guesses   = torch.argmax( outputs, 1)   # most probable guess per image

#         num_guess = len( guesses )              # how many in this batch
#         num_right = torch.sum( labels == guesses ).item()  # how many correct

#         num_correct  += num_right               # track accuracy
#         num_attempts += num_guess

# print("Total test accuracy:", 100*num_correct/num_attempts,"%")




# # ########## Automation:
# # # Write a line to a file
# # open('My_File_write.txt', 'w').write('I wrote this line\n')
# # # with open('My_File_write.txt', 'w') as read_file:
# # #     read_file.write('I wrote this line\n')

# # # # Append a line to the end of a file
# # with open('My_File_write.txt', 'a') as read_file:
# #     read_file.write('I appended this line\n')

# # # Write the items of a list to a txt file
# # x_list = ["2"] + ["3"]                    # the same as x_list = ["2", "3"]
# # print(x_list)
# # with open('My_File_write.txt', 'w') as write_file:
# #     write_file.writelines(x_list)         # each item in x_list must be a string 


# # # # Read a file, copy its lines to another file:
# # ### Opening both files:
# # with open('My_File_write.txt', 'w') as write_to_file, open('My_File_read.txt', 'r') as read_from_file:
# #     ### Reading the lines from one of them:
# #     lines_read = read_from_file.readlines()
# #     print(lines_read)
# #     ### Writing them to the other
# #     write_to_file.writelines(lines_read)



# # # # Read a file, copy its contents to another file,
# # # # append 10 more lines, each on the form of "Additional Line i" where i is the index of the added line
# # ### Opening both files:
# # with open('My_File_write.txt', 'w') as write_to_file, open('My_File_read.txt', 'r') as read_from_file:
# #     ### Reading the lines from one of them:
# #     lines_read = read_from_file.readlines()
# #     print(lines_read)
# #     ### Writing them to the other
# #     write_to_file.writelines(lines_read + [f'Additional Line {i}\n' for i in range(1,11)])
# # ### Another implementation by writing the fixed part then appending the dynamic part:
# # # with open('My_File_write.txt', 'w') as write_to_file, open('My_File_read.txt', 'r') as read_from_file:
# # #     ### Reading the lines from one of them:
# # #     lines_read = read_from_file.readlines()
# # #     print(lines_read)
# # #     ### Writing them to the other
# # #     write_to_file.writelines(lines_read)
# # # with open('My_File_write.txt', 'a') as append_to_file:
# # #     for i in range(1,11):
# # #         append_to_file.write(f"Additional Line {i}\n")




# # Xinv2 b c inv M=fan**1
# # Xinv3 c d inv M=fan**2
# # Xinv4 d e inv M=fan**3
# # Xinv5 e f inv M=fan**4
# # Xinv6 f g inv M=fan**5
# # Xinv7 g h inv M=fan**6
# # Xinv8 h i inv M=fan**7
# # Xinv9 i j inv M=fan**8
# # Xinv10 j k inv M=fan**9
# # Xinv11 k z inv M=fan**10