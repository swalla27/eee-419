# Example turning text into numbers usable by learning algorithms
# author: d updates by sdm

import numpy as np                            # needed for arrays

# turns words into frequency numbers
from sklearn.feature_extraction.text import CountVectorizer

# import the token transformer
from sklearn.feature_extraction.text import TfidfTransformer

# create an array of strings to study...
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining,the weather is sweet, and one and one is two'])

count = CountVectorizer()                     # create the vectorizer
bag = count.fit_transform(docs)               # now do the counting

# NOTE: order in dictionary is not necessarily the order in which words appear!
print('count ',count.vocabulary_)             # dictionary: key: number of times

# Each position corresponds to the number in the dictionary above...
print('count.fit_transform\n',bag.toarray())  # row per doc; column per token

np.set_printoptions(precision=2)              # set number of decimal places

tfidf = TfidfTransformer()                    # create the TFID calculator

# note values in each vector are such that the sum of the squares is ~1.00
print(tfidf.fit_transform(bag).toarray())     # do the calculation
