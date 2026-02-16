# Full sentimnent analysis of reviews
# author: d updated by sdm

import re                                                 # regular expressions
import pandas as pd                                       # read into data frame
from nltk.stem.porter import PorterStemmer                # get word stubs
from nltk.corpus import stopwords                         # meaningless words
from sklearn.model_selection import GridSearchCV          # parallel processing
from sklearn.pipeline import Pipeline                     # create tool chain
from sklearn.linear_model import LogisticRegression       # learning algorithm
from sklearn.feature_extraction.text import TfidfVectorizer  # words to numbers

################################################################################
# Function to remove HTML, put emoticons at end, change to lower case          #
# Input:                                                                       #
#    text - the string to be analyzed                                          #
# Output:                                                                      #
#    modified text                                                             #
################################################################################

def preprocessor(text):
    text = re.sub('<[^>]*>','',text)              # strip the html

    # extract the various emoticons that we expect
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)

    # convert to lower case then replace all nonword chars with whitespace
    # following this, append the emoticons after stripping the nose
    text = re.sub('[\W]+',' ',text.lower()) + ' ' + \
                  ' '.join(emoticons).replace('-','')

    return text    # and return the modified string
   
################################################################################
# Function to split a string into individual words                             #
# Input:                                                                       #
#    text - a string to be split                                               #
# Output:                                                                      #
#    a list of words or other elements                                         #
################################################################################

def tokenizer(text):
    return text.split()

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

df = pd.read_csv('./m6_4_1_movie_reviews.csv', encoding='latin-1') # read the data
print(df.head(3))                                           # check 1st entries

porter = PorterStemmer()             # create a stemmer

stop = stopwords.words('english')    # get the stop words

#print(preprocessor(df.loc[0 , 'review'][-50:]))  # debug example

df['review']=df['review'].apply(preprocessor)     # preprocess the data

# estimated 10+ hours to run 25000 ; split the train and test data
N=1000
X_train = df.loc[:N, 'review'].values
y_train = df.loc[:N, 'sentiment'].values
X_test = df.loc[N:2*N, 'review'].values
y_test = df.loc[N:2*N, 'sentiment'].values

# convert the text to numbers -> features!
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

# create a grid to find the optimal set of parameters
param_grid = [{'vect__ngram_range':[(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer':[tokenizer,tokenizer_porter],
               'clf__penalty':['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range':[(1,1)],
               'vect__stop_words': [stop,None],
               'vect__tokenizer':[tokenizer,tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty':['l1','l2'],
               'clf__C':[1.0, 10.0, 100.0]}
             ]

# create a pipeline to efficiently run through iterations
lr_tfidf = Pipeline([('vect',tfidf),
                     ('clf',
                      LogisticRegression(solver='liblinear',
                      random_state=0))])

# Create the actual grid search object and then do the fit
# The CV is for Cross Validation, splitting the training data 5 ways
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy',cv=5,
                           verbose=1, n_jobs=4)
gs_lr_tfidf.fit(X_train,y_train)

print('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
