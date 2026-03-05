import numpy as np
import pandas as pd
import time
import sys
import os

# ############### Review Topics:
# ###### M1:
# ### Conditional Statements
# # Question: Add a single character to this code to make it work:
# my_val = -2
# if ( my_val > 0 ):
#     print("positive")
# elif my_val > -5:
#     print("Negative > -5")
# else:
#     print("Too small")
# ### Loops
# # Q1: Write a command that imports a package to make this code work
# # Q2: How many iterations will this code run?
# from numpy import arange
# for index in arange(0,1,.1):
#     print(index)
# ### Lists
# # Q: What will be printed when this script runs?
# my_list_1 = [1 , 2 , 3]
# my_list_2 = [4 , 5]
# my_list_1.append(my_list_2)
# print(my_list_1)
# ### Functions
# # Q1: What error does this code have?
# # Q2: when you fix the error, what is the output of the print?
# def my_fun():
#     my_int = -5
# my_int = my_fun()
# print(my_int)
# # Q1: What command would you insert in my_fun so that it prints -5?
# def my_fun():
#     global my_int
#     my_int = -5
# my_fun()
# print(my_int)
### Printing Precision and Formatting
##### M2:
### Arrays
# Q: Write a command that creates an array full of 4 zeros
# import numpy as np
# array_bin_0 = np.zeros(4)
# print(array_bin_0)
#Q: What is the output of the following script:
# import numpy as np
# array_1 = np.full(4,1)
# array_2 = array_1
# array_2[0] = 1000
# print(array_1[0])
# # Q: Which of the following commands results in creating a copy of the array_1 into array_2?
# import numpy as np
# array_1 = np.full(4,1)
# array_2 = array_1
# # array_2 = array_1.copy() # Choice 1
# print(array_2)
# array_2 = np.copy(array_1) # Choice 2
# print(array_2)
# # array_2.copy(array_1) # Choice 3
# array_2[0] = 1000
# print(array_1[0])
# # Q: Which of the following commands results in creating a copy of the array_1 into array_2?
# from numpy import full
# array_1 = full(4,1)
# array_2 = array_1
# # array_2 = array_1.copy() # Choice 1
# array_2 = np.copy(array_1) # Choice 2
# # array_2.copy(array_1) # Choice 3
# array_2[0] = 1000
# print(array_1[0])
# ### Documentation and Try
# ### Strings
# ### Dictionaries
# # Q: Write a command that prints the value of the key "hat"?
# my_dict = {}
# my_dict["shirt"] = "polo"
# my_dict["hat"] = "fedora"
# # print(my_dict['hat'])
# # # Q (alternative): Which of the following prints the value of the key "hat"?
# print(my_dict["hat"]) # Choice 1
# # print(my_dict("hat")) # Choice 2
# # Q: What is the output of the following script
# my_dict = {}
# my_dict["shirt"] = "polo"
# my_dict["hat"] = "fedora"
# print(list(my_dict.values()))
# ### Plots
# # Q: T/F: The following script displays a plot with 1 curve?
import numpy as np # get the array functions
import matplotlib.pyplot as plt # get plotting functions
func_0 = np.zeros(10,int)
x_val = np.arange(0,10,1)
plt.plot(x_val,x_val)
plt.savefig('my_graph.png')
plt.show()
# ### Integration
# ##### M3:
# ### Linear Algebra
# Moc Midterm:
# #
# y = 10
# z = 3
# x = y//z
# print(x)
# stuff = [1, 27, "car"]
# x = stuff.pop(1)
# print(stuff)
# value = int(input("enter a number: "))
# loop_cnt = 0;
# while value != 1:
# loop_cnt += 1
# if ( value % 2 ) == 0:
# value //= 2
# else:
# value = ( value * 3 ) + 1
# print(" ", value)
# print(loop_cnt, "iterations")
# str_a = "happened"
# str_b = 'way'
# print("A funny thing",str_a,"on the"+str_b+"to the opera")
# import numpy as np
# # my_array = np.array([[1 , 2 , 3] , [4 , 5 , 6] , [7,8,9] , [4 , 5 ,8]])
# # print(len(my_array))
# print(np.full((3,5),7))
# print(np.ones((3, 5)) * 7)
# ### Solving an Electric Circuit
# ### Spring Analysis
# ##### M4:
# ### Signal Analysis
# ### GUI
# ### Loan Calculator
# ##### M5:
# ### ML
# ### Datasets and Correlation
# ### Perceptron and Logistic Regression
# ### KNN
# ##### M6:
# ### Reglar Expressions:
# # [] used to include multiple alternatives for the search
# # ^ Outside square brackets (^ at the beginning of the regex): Match thestart of the string
# # [^] Inside square brackets ([^...]): Match any character that is not listed in the brackets
# # + any one or more
# # [+] matches a plus sign
# # ? following a character (but outside []) means 0 or 1 occurance of this character
# # [?] matches a regular ?
# # . matches any character except for \n
# # [.] matches a regular dot
# import re # get the regular expression module
# ###### Question mark ? matches 0 or 1 of what immediately precedes it:
# str6 = 'here is happy :-) and here is sad :( and here is sarcastic :-P. Another :o), fifth one :a) 6th :aaa) and 7th :)'
# pat = re.search('[:;=][-]?[\(\)DP]',str6)
# print(pat.group())
# pats = re.findall('[:;=][-a]?[\(\)DP]',str6) # find them all
# # pats = re.findall('[:;=][-]?[\(\)DP]',str6)
# # pats = re.findall('[:;=][a]?[\(\)DP]',str6)
# # pats = re.findall('[:;=][-a?][\(\)DP]',str6) # don't put ? inside []
# # pats = re.findall('[:;=][-ao]?[\(\)DP]',str6)
# # pats = re.findall('[:;=][a-o]?[\(\)DP]',str6) # dash between characters represents range
# # pats = re.findall('[:;=].?[\(\)DP]',str6) # dot matches everything except \n
# # pats = re.findall('[:;=].+[\(\)DP]',str6) # + matches 1 or more (greedy)
# # pats = re.findall('[:;=].+?[\(\)DP]',str6) # +? matches 1 or more (non-greedy)
# # pats = re.findall('[:;=].*?[\(\)DP]',str6) # *? matches 0 or more (non-greedy)
# # pats = re.findall('[:;=].*[\(\)DP]',str6) # * matches 0 or more (greedy)
# # pats = re.findall('[:;=]\w+[\(\)DP]',str6) # \w matches a word character
# # pats = re.findall('[:;=]\W+[\(\)DP]',str6) # \W matches a non-word character
# print(pats)
