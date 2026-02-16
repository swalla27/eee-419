# Full sentimnent analysis of reviews
# author: d updated by sdm

from nltk.stem.porter import PorterStemmer                # get word stubs
from nltk.corpus import stopwords                         # meaningless words

################################################################################
# function to perform stemming on a string                                     #
# Input:                                                                       #
#    text - the string to stem                                                 #
# Output:                                                                      #
#    the stemmed and split string                                              #
################################################################################

def tokenizer_porter(text):
    return[porter.stem(word) for word in text.split()]

# start the main code here...

porter = PorterStemmer()             # create a stemmer

stop = stopwords.words('english')    # get the stop words

porter_check = [w for w in \
                tokenizer_porter('a runner likes running and runs a lot')\
                if w not in stop]
print("\n",porter_check,"\n")
