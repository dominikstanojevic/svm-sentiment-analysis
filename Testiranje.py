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

import pandas as pd
import numpy as np

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

df = pd.read_csv('./movie_data.csv', encoding='utf-8')

end = 25000

X_train = df.loc[:end, 'review'].values
y_train = df.loc[:end, 'sentiment'].values
X_test = df.loc[end:, 'review'].values
y_test = df.loc[end:, 'sentiment'].values

from nltk.corpus import stopwords
stop = stopwords.words('english') + list(punctuation)

from nltk.stem.porter import *
stemmer = PorterStemmer()


from Tokenizer import Tokenizer
Tok = Tokenizer(True, False, False)

def tok(text):
    tokens = Tok.tokenize(text)
    return negate(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(tokenizer=tok, stop_words=stop, max_features=20000, ngram_range=(1,1))
in_train = v.fit_transform(X_train).toarray()
in_test = v.transform(X_test).toarray()


# from sklearn.feature_extraction.text import CountVectorizer
# v = CountVectorizer(tokenizer=tok, stop_words=stop, max_features=5000, ngram_range=(1,1))
# in_train = v.fit_transform(X_train).toarray()
# in_test = v.transform(X_test).toarray()


from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(in_train, y_train)
print(svm.score(in_test, y_test))

'''weights = zip(v.get_feature_names(), np.nditer(svm.coef_))
sort_weights = sorted(weights, key = lambda x: np.abs(x[1]))
rank = len(sort_weights)
pos = 0
neg = 0
neutral = 0
for word, weight in sort_weights:
    print("Rank: " + str(rank) + ", Word: " + word + ", weight: " + str(weight))
    rank -= 1
    if np.isclose(weight, 0):
        neutral += 1
    elif weight < 0:
        neg += 1
    else:
        pos += 1
print(svm.intercept_)
print("Positive: " + str(pos) + ", negative: " + str(neg) + ", neutral: " + str(neutral))'''

flat = np.ndarray.flatten(svm.coef_)
predicted = svm.predict(in_test)
indices = np.where(predicted != y_test)
wrong = [(index, in_test[index, :].dot(flat) + float(svm.intercept_)) for index in np.nditer(indices)]
sorted_wrong = sorted(wrong, key = lambda x: np.abs(x[1]))

pos = 0
neg = 0
for index, value in sorted_wrong:
    print("Index: " + str(index) + ", value: " + str(value))
    if value > 0:
        pos += 1
    else:
        neg += 1
print("False positives: " + str(pos))
print("Flase negatives: " + str(neg))