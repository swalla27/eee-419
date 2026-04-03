# ## Basics of OOP:
# import numpy as np
# x_int = 2           # integer
# x_arr = np.array([[1, 2, 3],
#                   [4, 5, 6]])
# x_str = 'xyz'

# # Defining students in a school:
# # ----- WITHOUT OOP -----

# # Each student is a dictionary
# student1 = {"name": "Alice", "grade": 90, "age": 20}
# student2 = {"name": "Bob",   "grade": 85, "age": 21}

# # A function that prints a student
# def print_student(student):
#     print("Name:", student["name"])
#     print("Grade:", student["grade"])
#     print("Age:", student["age"])
#     print("--------------")

# # A function that boosts a student's grade
# def boost_grade(student, amount):
#     student["grade"] += amount

# # ----- TESTING -----
# print("WITHOUT OOP\n")
# print_student(student1)
# print_student(student2)

# boost_grade(student1, 5)
# print("After boosting Alice's grade:")
# print_student(student1)


# # ----- WITH OOP -----

# class student:
#     def __init__(self, namee, gradee, agee):
#         self.name = namee
#         self.grade = gradee
#         self.age = agee

#     def print_info(self):
#         print("Name:", self.name)
#         print("Grade:", self.grade)
#         print("Age:", self.age)
#         print("--------------")

#     def boost_grade(self, amount):
#         self.grade += amount


# # ----- TESTING -----
# print("\nWITH OOP\n")

# s1 = student("Alice", 90, 20)
# s2 = student("Bob",   85, 21)

# s1.print_info()
# s2.print_info()

# s1.boost_grade(5)
# print("After boosting Alice's grade:")
# s1.print_info()
# print(type(s1))



# # =====================================================================
# # SUBCLASSES (Inheritance, used when classes have different attributes)
# # =====================================================================

# class dog:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def speak(self):
#         return "Woof!"


# class cat:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def speak(self):
#         return "Meow!"


# class animal:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     # def speak(self):
#     #     return "A Default Animal Sound!"

# class dog(animal):
#     # def __init__(self, name, age):#, breed):
#     #     super().__init__(name, age)
#     #     # self.breed = breed

#     def speak(self):
#         return "Woof!"


# class cat(animal):
#     def speak(self):
#         return "Meow!"


# # ----- TESTING -----
# d = dog("Rex", 4)
# c = cat("Whiskers", 5)

# print(d.name, "says:", d.speak())
# print(c.name, "says:", c.speak())


# # ========================================
# # Subclasses Another example:
# # author: sdm
# # ========================================

# PI = 3.14                              # define for convenience

# class shape:                           # create the class
#     def __init__(self,name):           # called when object is created
#         self.name = name               # instance variables
#         self.area = 0.0
#         self.perimeter = 0.0

#     def print_prop(self):                   # print the values
#         print(self.name,"has perimeter",self.perimeter,
#               'and area',self.area)

# class square(shape):                   # parent class in parentheses
#     def __init__(self,side):
#         shape.__init__(self,'square')
#         self.side = side
#         self._calcp_()
#         self._area_()

#     def _calcp_(self):                 # method to calculate perimeter
#         self.perimeter = 4 * self.side

#     def _area_(self):                 # method to calculate area
#         self.area = self.side * self.side

#     def update(self,side):             # update the side length
#         self.side = side
#         self._calcp_()
#         self._area_()

# class circle(shape):                   # parent class in parentheses
#     def __init__(self,radius):
#         shape.__init__(self,'circle')
#         self.radius = radius
#         self._calcp_()
#         self._area_()

#     def _calcp_(self):                 # method to calculate perimeter
#         self.perimeter = 2 * PI * self.radius

#     def _area_(self):                 # method to calculate area
#         self.area = PI * self.radius * self.radius

#     def update(self,radius):          # update the radius length
#         self.radius = radius
#         self._calcp_()
#         self._area_()

# sq2 = square(2)
# sq4 = square(4)

# sq2.print_prop()
# sq4.print_prop()

# sq2.update(5)
# sq2.print_prop()

# cir2 = circle(2)
# cir4 = circle(4)

# cir2.print_prop()
# cir4.print_prop()

# cir2.update(8)
# cir2.print_prop()



# # =================
# # Machine Learning:
# # =================
# # author hbm updated by sdm

# import torch                         # import the various PyTorch packages
# import torch.nn            as nn
# import torch.optim         as optim
# import torch.nn.functional as F
# import numpy as np

# from torchvision import datasets     # the data repository
# from torchvision import transforms   # tranforming the data

# ################################################
# # Setting up constants for training
# ################################################
# BATCH_SIZE  = 128   # number of samples per gradient update
# NUM_CLASSES = 10    # how many classes to classify (10 digits, 0-9)
# EPOCHS      = 2     # how many epochs to run trying to improve

# ################################################
# # Create the network
# ################################################
# # for i in dir(nn):
# #     print(i)
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
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                           shuffle=True)
# testset     = datasets.MNIST('~/MNIST_data/test', download=True,
#                              train=False, transform=transform)
# testloader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                           shuffle=True)

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
