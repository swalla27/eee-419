# Small emoticon example
# author: d updates by sdm

import re                       # get regular expression module
import pandas as pd             # get pandas to read the csv file

################################################################################
# Function to preprocess text                                                  #
# Input:                                                                       #
#    text - the text to be processed                                           #
# Output:                                                                      #
#    the processed text is returned                                            #
################################################################################

def preprocessor(text):
    print("\nWorking on:\n"+text)             # show text being worked on

    # get rid of HTML, which is of the form <.../> where ... are any non '>'
    text = re.sub('<[^>]*>','',text)
    print(text)

    # now find the emoticons
    emoticons = re.findall('[:;=][-]?[\)\(DP]',text)
    print(emoticons)

    # now get rid of any nonword characters, change to lowercase,
    # and append the emoticons
    # NOTE: the \ at the end of the line is a continuation character
    # NOTE: the code for this in the book is missing the space!
    text = re.sub('[\W]+',' ',text.lower()) + ' ' + \
                  ' '.join(emoticons).replace('-','')
    print(text+"\n\n")    # separate the strings!
    return text
   
df = pd.read_csv('m6_3_1_preprocess.csv', encoding='latin-1')     # read the file
print(df.head(3))                                   # print the first 3 lines

# run on first entry as debug...
#print(preprocessor(df.loc[0 , 'review']))

# apply the preprocessor function to each entry, one at a time...
df['review']=df['review'].apply(preprocessor)

   
