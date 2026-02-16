# import numpy as np
# import matplotlib.pyplot as plt

# NUM_PTS = 100
# x = np.linspace(0.01,2,NUM_PTS)
# y = 0.5*x + 0.01*np.random.randn(NUM_PTS)

# ### Correlation Coefficient is a measure of how much linear two variables are:
# corr_coeff = np.corrcoef(x,y)
# print(corr_coeff)

# plt.scatter(x,y)
# plt.ylim(0, 1)
# plt.show()



# ####### Use Principle Component Analysis (PCA):
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.decomposition import PCA                  # PCA package


# NUM_PTS = 21
# #### x1 and x2 are the features (inputs) of the dataset, y is the class (output)
# x1 = np.linspace(-10, 10, NUM_PTS)
# x2 = 2 * np.ones_like(x1) + 0.01*np.random.randn(NUM_PTS)
# y = np.where(x1 >= 0, 1, 0)

# plt.scatter(x1,x2)
# plt.show()
# ### convert to df and print for visualization:
# df = pd.DataFrame({'x1': x1 , 'x2': x2 , 'y': y})
# print(df)

# features_mat = np.column_stack((x1,x2))
# print("features matrix =\n", features_mat)

# ### A matrix that selects the first colum
# select_mat = np.array([[1] , 
#                        [0]])
# print("\nSelecting Matrix =\n",select_mat)

# ### dot product to do the selection
# new_feat_mat = np.dot(features_mat,select_mat)
# print("\nMatrix w Selected Features only=\n",new_feat_mat)

# ### Let's Use Principle Component Analysis:
# pca = PCA(n_components=2)
# new_feat_mat = pca.fit_transform(features_mat) # apply to the train data

# print("\nNew Features w PCA =\n",new_feat_mat)

# # Eigenvalues (variance explained by each principal component)
# eigenvalues = pca.explained_variance_
# print("Eigenvalues:", eigenvalues)

# # Proportion of variance explained (normalized)
# explained_var_ratio = pca.explained_variance_ratio_
# print("Explained variance ratio:", explained_var_ratio)

# # Cumulative variance (to decide # of components)
# cumulative_var = np.cumsum(explained_var_ratio)
# print("Cumulative explained variance:", cumulative_var)







# ###### Three Features:
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from numpy.linalg import eig   # import eigenvalue function
# from sklearn.decomposition import PCA                  # PCA package

# NUM_PTS = 21
# #### x1 and x2 are the features (inputs) of the dataset, y is the class (output)
# x1 = np.linspace(-10, 10, NUM_PTS)
# x2 = np.linspace(-5, 5, NUM_PTS)
# x3 = np.linspace(50, 500, NUM_PTS)
# y = np.where(x1 >= 0, 1, 0)

# # plt.scatter(x1,x2)
# # plt.show()

# ### convert to df and print for visualization:
# df = pd.DataFrame({'x1': x1 , 'x2': x2 , 'x3': x3 , 'y': y})
# print(df)

# features_mat = np.column_stack((x1,x2,x3))
# # features_mat = features_mat + 0.001*np.random.randn(NUM_PTS,2)
# print("features matrix =\n", features_mat)

# ### Let's Use Principle Component Analysis:
# pca = PCA(n_components=3)                    # only keep one (the best) feature!
# new_feat_mat = pca.fit_transform(features_mat) # apply to the train data

# print("\nNew Features w PCA =\n",new_feat_mat)

# # Eigenvalues (variance explained by each principal component)
# eigenvalues = pca.explained_variance_
# print("Eigenvalues:", eigenvalues)

# # Proportion of variance explained (normalized)
# explained_var_ratio = pca.explained_variance_ratio_
# print("Explained variance ratio:", explained_var_ratio)

# # Cumulative variance (to decide # of components)
# cumulative_var = np.cumsum(explained_var_ratio)
# print("Cumulative explained variance:", cumulative_var)


#### Alternatively, we can do PCA manually by performing eigen analysis:
# # # find both eigenvalues and eigenvector
# eig_val , eig_vec = eig(features_mat)
# eig_val = np.real(eig_val)
# eig_vec = np.real(eig_vec)
# print("\nEigen Values =\n",eig_val)
# print("\nEigen Vectors =\n", eig_vec)

# ### A matrix that selects the first colum
# rotate_mat = eig_vec
# print("\nSelecting Matrix =\n",rotate_mat)

# ### dot product to do the selection
# new_feat_mat = np.dot(features_mat,rotate_mat)
# print("\nMatrix after rotation=\n",new_feat_mat)



# #####################################
# ################# EFFECT of PCA:
# ######## First: Without PCA:


# import numpy as np                                     # needed for arrays
# import pandas as pd                                    # data frame
# import matplotlib.pyplot as plt                        # modifying plot
# from sklearn.model_selection import train_test_split   # splitting data
# from sklearn.preprocessing import StandardScaler       # scaling data
# from sklearn.linear_model import LogisticRegression    # learning algorithm
# from sklearn.decomposition import PCA                  # PCA package
# from sklearn.metrics import accuracy_score             # grading
# from sklearn.metrics import confusion_matrix           # generate the  matrix
# # from m5_3_3_plotdr import plot_decision_regions            # fancy plot

# # read the database. Since it lacks headers, put them in
# df_wine = pd.read_csv('m6_1_1_wine.csv',header=None)
# df_wine.columns = ['class label','alcohol','malic acid','ash',
#                    'alcalinity of ash','magnesium','total phenols','flavanoids',
#                    'nonflavanoid phenols','proanthocyanins','color intensity',
#                    'hue','od280/0d315 of diluted wines','proline']

# # list out the labels
# print('Class labels', np.unique(df_wine['class label']))

# print(df_wine)
# X = df_wine.iloc[:,1:].values       # features are in columns 1:(N-1)
# y = df_wine.iloc[:,0].values        # classes are in column 0!

# # now split the data
# X_train, X_test, y_train, y_test = \
#          train_test_split(X, y, test_size=0.3, random_state=0)

# stdsc = StandardScaler()                     # apply standardization
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.transform(X_test) 

# # Look at the sorted correlation coeff. matrix
# corr_coeff_wine = df_wine.corr().abs()
# corr_coeff_wine *= np.tri(*corr_coeff_wine.values.shape, k=-1).T
# print(corr_coeff_wine)
# corr_unstack = corr_coeff_wine.unstack().copy()
# # Sort values in descending order
# corr_unstack.sort_values(inplace=True,ascending=False)
# pd.set_option('display.max_rows', None)
# print(corr_unstack)
# # print((df_wine))
# # input()
# # NOTE: only keep two features as that's all plot_decision_regions can handle!

# X_train = X_train_std[: , [7 , 12]] # apply to the train data
# X_test = X_test_std[: , [7 , 12]]       # do the same to the test data

# # now create a Logistic Regression and train on it
# lr = LogisticRegression(solver='liblinear', multi_class='ovr')
# lr.fit(X_train,y_train)

# y_pred = lr.predict(X_test)              # how do we do on the test data?
# print('Number in test ',len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy:', accuracy_score(y_test, y_pred))

# # # now combine the train and test data and see how we do
# # X_comb = np.vstack((X_train, X_test))
# # y_comb = np.hstack((y_train, y_test))
# # print('Number in combined ',len(y_comb))
# # y_comb_pred = lr.predict(X_comb)
# # print('Misclassified combined samples: %d' % (y_comb != y_comb_pred).sum())
# # print('Combined Accuracy: %.2f' % accuracy_score(y_comb, y_comb_pred))

# # confuse = confusion_matrix(y_comb,y_comb_pred)
# # print(confuse)

# # # # Now visualize the results
# # # plot_decision_regions(X_train, y_train, classifier=lr)
# # # plt.xlabel('PC 1')
# # # plt.ylabel('PC 2')
# # # plt.legend(loc='lower left')
# # # plt.title('wine analysis with two components')
# # # plt.show()


# ################################
# # # Second: With PCA:

# import numpy as np                                     # needed for arrays
# import pandas as pd                                    # data frame
# import matplotlib.pyplot as plt                        # modifying plot
# from sklearn.model_selection import train_test_split   # splitting data
# from sklearn.preprocessing import StandardScaler       # scaling data
# from sklearn.linear_model import LogisticRegression    # learning algorithm
# from sklearn.decomposition import PCA                  # PCA package
# from sklearn.metrics import accuracy_score             # grading
# from sklearn.metrics import confusion_matrix           # generate the  matrix
# from m5_3_3_plotdr import plot_decision_regions            # fancy plot

# # read the database. Since it lacks headers, put them in
# df_wine = pd.read_csv('m6_1_1_wine.csv',header=None)
# df_wine.columns = ['class label','alcohol','malic acid','ash',
#                    'alcalinity of ash','magnesium','total phenols','flavanoids',
#                    'nonflavanoid phenols','proanthocyanins','color intensity',
#                    'hue','od280/0d315 of diluted wines','proline']

# # list out the labels
# print('Class labels', np.unique(df_wine['class label']))

# X = df_wine.iloc[:,1:].values       # features are in columns 1:(N-1)
# y = df_wine.iloc[:,0].values        # classes are in column 0!

# # now split the data
# X_train, X_test, y_train, y_test = \
#          train_test_split(X, y, test_size=0.3, random_state=0)

# stdsc = StandardScaler()                     # apply standardization
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.transform(X_test) 

# # NOTE: only keep two features as that's all plot_decision_regions can handle!

# pca = PCA(n_components=2)                    # only keep two "best" features!
# X_train_pca = pca.fit_transform(X_train_std) # apply to the train data
# X_test_pca = pca.transform(X_test_std)       # do the same to the test data

# # # Eigenvalues (variance explained by each principal component)
# # eigenvalues = pca.explained_variance_
# # print("Eigenvalues:", eigenvalues)

# # # Proportion of variance explained (normalized)
# # explained_var_ratio = pca.explained_variance_ratio_
# # print("Explained variance ratio:", explained_var_ratio)

# # # Cumulative variance (to decide # of components)
# # cumulative_var = np.cumsum(explained_var_ratio)
# # print("Cumulative explained variance:", cumulative_var)

# # now create a Logistic Regression and train on it
# lr = LogisticRegression(solver='liblinear', multi_class='ovr')
# lr.fit(X_train_pca,y_train)

# y_pred = lr.predict(X_test_pca)              # how do we do on the test data?
# print('Number in test ',len(y_test))
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# # now combine the train and test data and see how we do
# X_comb_pca = np.vstack((X_train_pca, X_test_pca))
# y_comb = np.hstack((y_train, y_test))
# print('Number in combined ',len(y_comb))
# y_comb_pred = lr.predict(X_comb_pca)
# print('Misclassified combined samples: %d' % (y_comb != y_comb_pred).sum())
# print('Combined Accuracy: %.2f' % accuracy_score(y_comb, y_comb_pred))

# confuse = confusion_matrix(y_comb,y_comb_pred)
# print(confuse)

# # Now visualize the results
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.title('wine analysis with two components')
# plt.show()


######################
##### Regular Expressions:

import re                               # get the regular expression module

# str1 = 'what a fine day this is'        # create a string
# print(re.split(' ',str1))            # splits scentence based on letter 'a'. Note 'aa' has no letter in between so '' shows in the output
# print(re.split('a',str1))           # splits on one or more occurance of 'a'. Hence, '' does not show up in the output

# # str5 = 'EEE (419)'
# # print(str5)
# str6 = 'here is happy :-) and here is sad :(  :-P.  :o),  :a), :aaa), :), ;)  :)  :b) :\\ This is a review <silly/> with :-) not at the end'
# pats = re.findall('[<]*[>]',str6)                         # finds only the first occurance
# # pat = re.search(':\)',str6)                         # finds only the first occurance
# print(pat.group())
# pats = re.findall(':\)',str6)                       # finds them all
# print(pats)
# pats = re.findall(':[\(\)]',str6)           
# print(pats)
# pats = re.findall('[:;][\(\))]',str6)          
# print(pats)
# pats = re.findall('[:;=][-][\(\)DP]',str6)           
# print(pats)
# pats = re.findall('[:;=]-[\(\)DP]',str6)            # finds them all
# print(pats)
# pats = re.findall('[:;=][-]?[\(\)DP]',str6)
# print(pats)
# pats = re.findall('[:;=][-a]?[\(\)DP]',str6)
# print(pats)
# pats = re.findall('[:;=][a]?[\(\)DP]',str6)
# print(pats)
# pats = re.findall('[:;=][-a?][\(\)DP]',str6)         # don't put ? inside [], unless you want to match a '?'
# print(pats)
# pats = re.findall('[:;=][a-o][\(\)DP]',str6)         # dash between characters represents range
# print(pats)
# pats = re.findall('[:;=][-ao][\(\)DP]',str6)
# print(pats)
# pats = re.findall('[:;=].[\(\)DP]',str6)             # dot matches everything except \n
# print(pats)
# pats = re.findall('[:;=][.][\(\)DP]',str6)           # but don't put the dot in []
# print(pats)
# pats = re.findall('[:;=]+[\(\)DP]',str6)            #  + matches 1 or more (greedy)
# print(pats)
# pats = re.findall('[:;=][a-z]+[\(\)DP]',str6)            #  + matches 1 or more (greedy)
# print(pats)
# str6 = 'here is happy :-) and here is sad :(  :-P.  :o),  :a), :aaa), :), ;)  :)  :b)'
# pats = re.findall('[:;=].+[\(\)DP]',str6)            #  + matches 1 or more (greedy)
# print(pats)
# pats = re.findall('[:;=]a*[\(\)DP]',str6)            #  * matches 0 or more (greedy)
# print(pats)
# pats = re.findall('[:;=].*[\(\)DP]',str6)            #  * matches 0 or more (greedy)
# print(pats)
# pats = re.findall('[:;=].+?[\(\)DP]',str6)           #  +? matches 1 or more (non-greedy)
# print(pats)
# pats = re.findall('[:;=].*?[\(\)DP]',str6)           #  *? matches 0 or more (non-greedy)
# print(pats)
# pats = re.findall('([:;=])(\w+)[\(\)DP]',str6)           #  \w matches a word character
# print(pats)
# pats = re.findall('[:;=]\W+[\(\)DP]',str6)           #  \W matches a non-word character
# print(pats)

# text = "SDasd John Doe, 25 years old"
# # match = re.findall("\w+", text)
# # print(match)
# # # # match = re.search(r"(\w+)", text)
# match = re.search(r"(\w+) (\w+), (\d+) years old", text)
# print("First Name:", match.group(1))
# print("Last Name:", match.group(2))
# print("Age:", match.group(3))
# print("Entire Pattern:", match.group())


# quote = 'this is "start of stuff "now more stuff" and even "more stuff" final things" and some other stuff'

# # first, extract the inner quote using a greedy search
# first = re.search('"(.*)"',quote)
# print(first.group())
# print(first.group(1))

# matched_text = re.findall('"(.*)"',quote)
# print(matched_text[1][1:-1])


# # can do substitutions!
# # re.sub(pattern, replacement, str)
# str7 = 'this string has a typoo typoo'                  # bad text
# # new_str = re.sub('[o]+','o',str7)                 # fix it!
# new_str = re.sub('[o+]','o',str7)                 # doesn't fix it!
# print('it was:',str7,"\nbut now it's:",new_str)

#### Summary of Rules:
## [] used to include multiple alternatives for the search
## [^] Inside square brackets and at the beginning ([^...]): Match any character that is not listed in the brackets
## [...^] Inside square brackets but is not at the beginning (at the middle or end): Match a regular ^
## ^ Outside square brackets (^ at the beginning of the regex): Match the start of the string
## [?] matches a regular question mark "?"
## ? following a character (but outside []) means 0 or 1 occurance of this character
## [+] matches a plus sign
## + following a character (but outside []) means 1 or more occurance of this character
## * following a character (but outside []) means 0 or more occurance of this character
## [*] matches a regular *
## +? following a character acts the same as + but non-greedy (stops at the first occurance of the following character)
## *? following a character acts the same as * but non-greedy (stops at the first occurance of the following character)
## [.] matches a regular dot
## . matches any character except for \n
## \w matches a word character a-z, A-Z
## \W matches a non-word character
## $ asserts that the pattern must be at the end of the string
## (...) capturing group which: capcutres what's inside and stores it into a group that it is accessible via .group(1) .group(2)...etc.





# Exercises
# (1):
# Write a code to show that when a feature X2 is not correlated at all with the o/p y,
# building a ML algorithm without X2 gives a better accuracy than with it
# Hint 1: Let y = sign(X1), where X1 = 1 ----> 100 + noise1, while X2 = noise2
# Hint 2: make sure noise1 has small variance, while noise2 has high variance

# (2):
# Write a code to show that when two features X1 and X2 correlate highly with each other and with the o/p y,
# building a ML algorith with both features give close accuracy to an ML with only X1
# Hint 1: Let X = 1 ----> 100, X1=X+noise1, X2=X+noise2, y=+1 if X1 and X2 >= 0, and 0 otherwise
# Hint 2: make sure noise1 and noise2 have small variances

# (3):
# Re-solve exercise (1) while increasing the dataset size significantly. Write a code to show that 
# as you increase the dataset size, the accuracy with X2 approaches that without X2.
# Hint 1: if you cannot observe this property, you might want to work with a smaller variance of noise2
# Hint 2: if you still cannot observe it, you might want to, instead, fix the dataset size and observe the 
#         change in accuracy with the decrease in variance of noise2

# import re
# str9 = 'xxone and yyone and zzone'
# print(re.search('^..one', str9).group())