# Example of how to create a fake email box
# author: sdm

import pandas as pd                          # needed to read the data file
from numpy.random import choice              # needed to randomly pick names
from datetime import datetime                # needed to get the date and time

NUM_EMAILS = 1000                            # how many eamils to generate

################################################################################
# Function to read names and ranks and create a list with each name repeated   #
# the number of times corresponding to its rank.                               #
# inputs:  file_name - the name of the file to read with names and ranks       #
# outputs: name_list - the list of names to use as senders                     #
################################################################################

def read_data(file_name):
    name_list = []                           # create an empty list
    
    df = pd.read_csv(file_name)              # read the file
    names = df.iloc[:,0].values              # first column is the names
    ranks = df.iloc[:,1].values              # second column is the rank
    num_names = max(ranks) + 1               # compensate for highest rank=1

    #print(names)
    #print(ranks)

    # NOTE: the file contains the rank of the name, with 1 being highest
    # so we compensate for that by subtracting the rank from one more than
    # the lowest rank. That is, if the lowest rank is 24, we subtract from
    # 25 so the frequency in the name list is reversed: the name ranked #1
    # appears 25 times and the name ranked 24 appears once.

    for name, rank in zip(names,ranks):      # go through each name/rank
        #print(name,num_names-rank)
        name_list.extend([name]*(num_names-rank))  # append copies of the name

    return name_list

name_list = read_data("scottish names 2010.csv")
#print(name_list)

from_list = choice(name_list,NUM_EMAILS)     # pick names with replacement
#print(from_list)

mailbox = open("mailbox.txt","w")            # open the output file
for sender in from_list:
    now = datetime.now()                           # get current time/date
    now_str = now.strftime("%B %d, %Y %H:%M:%S")   # get the desired format

    mailbox.write("From "+sender+"@fakeco.fakeurl "+now_str+"\n")
    mailbox.write("Return-Path: <postmaster@fakeco.fakeurl>\n")
    mailbox.write("To: steve.millman@fakeco.fakeurl\n")
    mailbox.write("From: "+sender+"@fakeco.fakeurl\n")
    mailbox.write("Subject: Python Additional Topics\n")
    mailbox.write("\n")
    mailbox.write("I will recommend this class to all my friends!\n")
    mailbox.write("\n")

mailbox.close()    # don't forget to close the file!
