
# coding: utf-8

# In[2]:

from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

import nltk

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word[0], get_wordnet_pos(word[1])) for word in tags]


# In[ ]:

import re
import string
negation_words = {'never', 'no', 'nothing', 'nowhere', 'noone', 'none', 'not', 'havent', 'hasnt', 'hadnt', 'cant',
                  'couldnt', 'shouldnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'aint', 'n\'t', 
                  "haven't", "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't", "wouldn't", "don't", "didn't", 
                  "isn't", "aren't", "ain't", "neither", "nor"}

negate_re = re.compile(r'(' + '|'.join(negation_words) + ')', re.VERBOSE | re.I | re.UNICODE)

punctuation = set(string.punctuation)


def negate(tokens):
    words = list(tokens)
    negate_mod = False
    for i in range(len(words)):
        word = words[i]
        if negate_re.match(word):
            negate_mod = True
            continue
        elif word in punctuation:
            negate_mod = False
            continue
        if negate_mod:
            words[i] = word + "_NEG"
    return words


# In[ ]:

import pandas as pd
import numpy as np

df = pd.read_csv('./movie_data.csv', encoding='utf-8')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from nltk.corpus import stopwords
stop = stopwords.words('english') + list(punctuation)

from Tokenizer import Tokenizer
Tok = Tokenizer(True, False, False)
def tok(text):
    tokens = Tok.tokenize(text)
    words = lemmatize(tokens)
    return negate(words)
    

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(tokenizer=tok, stop_words=stop, max_features=20000, min_df=0.1, max_df=0.9)
X_train = v.fit_transform(X_train).toarray()
X_test = v.transform(X_test).toarray()


# In[ ]:

import SVM
svm = SVM.LinearSVM()
svm.fit(X_train, y_train)
svm.score(X_train, y_train)


# In[ ]:

file = open('test.txt', 'r')
test = file.read()
file.close()
print(svm.predict(v.transform([test]).toarray()))

