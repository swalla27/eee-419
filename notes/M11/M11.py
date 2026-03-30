# ################
# ##### Basics of Generating Random Numbers in Python:

# import numpy as np
# import random
# import time
# import math

# # # # Get the current Unix time
# unix_time = time.time()
# print("Current Unix Time:", unix_time)
# # # 1732319522.5896983
# lambdaa = 0.2
# np.random.seed(int(unix_time))
# x = np.random.uniform(size = 3)
# y = np.log(1-x)/(-lambdaa)
# print(x)
# print(y)




# ####################
# ####################
# ### Histogram:
# # 
# # # Uniform Random Variable:
# import matplotlib.pyplot as plt
# import numpy as np

# N = int(1e4)
# uniform_rv = np.random.uniform(size=N)
# plt.hist(uniform_rv, bins=10, density=True, edgecolor='black', label='Uniform')

# # Exponential RV:
# lambdaa = 1
# exp_rv = np.log(1-uniform_rv)/(-lambdaa)
# plt.hist(exp_rv, bins=1000, density=True, edgecolor='black', label= 'Exponential')

# plt.xlabel('Value')
# plt.ylabel('PDF')
# plt.title('Prob. Mass Func.')
# plt.legend()
# plt.show()


# ###############################
# ###############################
# ### Generating a normally-distributed random variable
# import matplotlib.pyplot as plt
# import numpy as np

# N = int(1e4)
# unif_1 = np.random.uniform(size=N)
# unif_2 = np.random.uniform(size=N)

# ## Generate a Rayleigh RV = sqrt(exp_rv)
# lambdaa = 0.5
# exp_rv = - np.log(1-unif_1)/(lambdaa)
# rayleigh_rv = np.sqrt(exp_rv)
# # plt.hist(rayleigh_rv, bins=100, density=True, edgecolor='black', label='Rayleigh')
# plt.hist(exp_rv, bins=100, density=True, edgecolor='black', label='Exp')

# ## Generate a uniformly distributed random variable theta over [-pi,pi):
# theta_rv = unif_2 * 2 * np.pi # (unif_2 * 2 - 1) * np.pi
# plt.hist(theta_rv, bins=100, density=True, edgecolor='black', label='Uniform')

# ## Generate a normally distributed RV:
# x_norm = rayleigh_rv * np.cos(theta_rv)
# # plt.hist(x_norm, bins=100, density=True, edgecolor='black', label='Normal')


# x_norm_built_in = np.random.normal(loc=0, scale=1, size=N)
# plt.hist(x_norm_built_in, bins=100, density=True, edgecolor='black', label='Normal Built-in')

# plt.xlabel('Value')
# plt.ylabel('Prob. Mass Func.')
# plt.title('Histogram')
# plt.legend()
# plt.show()




# # ##########################
# # ### ML: Classifying the distribution the data is drawn from:

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# NUM_FEATURES = 100
# NUM_PTS_PER_DIST = 100
# SIZE_PER_DIST = (NUM_PTS_PER_DIST,NUM_FEATURES)

# # Build three dataframes with class labels
# unif_rv = np.random.uniform(size=SIZE_PER_DIST)
# df_unif = pd.DataFrame(unif_rv, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
# df_unif["class"] = 0

# exp_rv = np.random.exponential(size=SIZE_PER_DIST)
# df_exp = pd.DataFrame(exp_rv, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
# df_exp["class"] = 1

# norm_rv = np.random.normal(loc=0, scale=1, size= SIZE_PER_DIST)
# df_norm = pd.DataFrame(norm_rv, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
# df_norm["class"] = 2

# # Concatenate into one dataset
# df = pd.concat([df_unif, df_exp, df_norm], ignore_index=True)

# # Split into features and target
# X = df.drop("class", axis=1)
# y = df["class"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# # Train a classifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_test)# Calculate accuracy
# accuracy = clf.score(X_test, y_test)
# print(f"Accuracy: {accuracy:.4f}")




# ##########################
# ### ML: What if all are normally distributed with different statistcs:

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# NUM_FEATURES = 5
# NUM_PTS_PER_DIST = 1000
# SIZE_PER_DIST = (NUM_PTS_PER_DIST,NUM_FEATURES)

# STD_DEV = 1
# norm_rv_0 = np.random.normal(loc=0, scale=STD_DEV, size= SIZE_PER_DIST)
# df_norm_0 = pd.DataFrame(norm_rv_0, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
# df_norm_0["class"] = 0

# norm_rv_1 = np.random.normal(loc=1, scale=STD_DEV, size= SIZE_PER_DIST)
# df_norm_1 = pd.DataFrame(norm_rv_1, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
# df_norm_1["class"] = 1

# norm_rv_2 = np.random.normal(loc=2, scale=STD_DEV, size= SIZE_PER_DIST)
# df_norm_2 = pd.DataFrame(norm_rv_2, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
# df_norm_2["class"] = 2

# # Concatenate into one dataset
# df = pd.concat([df_norm_0, df_norm_1, df_norm_2], ignore_index=True)

# # Split into features and target
# X = df.drop("class", axis=1)
# y = df["class"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# # Train a classifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_test)# Calculate accuracy
# accuracy = clf.score(X_test, y_test)
# print(f"Accuracy: {accuracy:.4f}")



# ###################################
# ######## Plotting vs NUM_PTS_PER_DIST
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# NUM_FEATURES = 5
# STD_DEV = 5
# accuracies = []

# NUM_PTS_VEC = [100, 1000, 10000, 100000]
# for NUM_PTS_PER_DIST in NUM_PTS_VEC:
#     SIZE_PER_DIST = (NUM_PTS_PER_DIST, NUM_FEATURES)
    
#     # Generate data for three classes
#     df_list = []
#     for class_id, loc in enumerate([0, 1, 2]):
#         data = np.random.normal(loc=loc, scale=STD_DEV, size=SIZE_PER_DIST)
#         df_tmp = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
#         df_tmp["class"] = class_id
#         df_list.append(df_tmp)
        
#     df = pd.concat(df_list, ignore_index=True)
    
#     # Split into features and target
#     X = df.drop("class", axis=1)
#     y = df["class"]
    
#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )
    
#     # Train classifier and calculate accuracy
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)
    
#     print(f"NUM_PTS_PER_DIST={NUM_PTS_PER_DIST}, Accuracy: {accuracy:.4f}")
#     accuracies.append(accuracy)

### accuracies = [0.3667, 0.4256, 0.4186, 0.4232]
# plt.plot(NUM_PTS_VEC, accuracies, marker='o')
# plt.xlabel('Number of Points per Distribution')
# plt.xscale('log')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Number of Points per Distribution')
# plt.grid(True)
# plt.show()





# #############################
# ######## Plotting vs NUM_FEATURES
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# NUM_PTS_PER_DIST = 1000  # Fixed number of points per distribution
# STD_DEV = 5
# accuracies = []

# NUM_FEATURES_VEC = [5, 10, 20, 50]
# for NUM_FEATURES in NUM_FEATURES_VEC:
#     SIZE_PER_DIST = (NUM_PTS_PER_DIST, NUM_FEATURES)
    
#     # Generate data for three classes
#     df_list = []
#     for class_id, loc in enumerate([0, 1, 2]):
#         data = np.random.normal(loc=loc, scale=STD_DEV, size=SIZE_PER_DIST)
#         df_tmp = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(NUM_FEATURES)])
#         df_tmp["class"] = class_id
#         df_list.append(df_tmp)
        
#     df = pd.concat(df_list, ignore_index=True)
    
#     # Split into features and target
#     X = df.drop("class", axis=1)
#     y = df["class"]
    
#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42, stratify=y
#     )
    
#     # Train classifier and calculate accuracy
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X_train, y_train)
#     accuracy = clf.score(X_test, y_test)
    
#     print(f"NUM_FEATURES={NUM_FEATURES}, Accuracy: {accuracy:.4f}")
#     accuracies.append(accuracy)

# plt.plot(NUM_FEATURES_VEC, accuracies, marker='o')
# plt.xlabel('Number of Features')
# plt.xscale('log')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Number of Features')
# plt.grid(True)
# plt.show()


# ############################
# ####### Code:

# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# N = 10
# # Example dataset (1 feature, 1 output with 10 classes)
# # Replace this with your actual dataset
# # X = np.random.rand(N, 1)  # 100 samples, 1 feature
# # y = np.random.randint(0, 10, N)  # Random labels from 0 to 9
# X = range(N)
# y = range
# print(X,y)
# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 nearest neighbors (can be adjusted)
# knn.fit(X_train, y_train)

# # Predict on the test set
# y_pred = knn.predict(X_test)

# # Evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print("Test Accuracy:", accuracy)
