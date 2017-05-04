import nltk
import string
import os
import sys, unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

path = '/opt/datacourse/data/parts'
token_dict = {}
stemmer = PorterStemmer()
punctuation = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

local_document = "It was remarked in the preceding paper, that weakness and divisions at home would invite dangers from abroad; and that nothing would tend more to secure us from them than union, strength, and good government within ourselves. This subject is copious and cannot easily be exhausted."


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def documents():
    docs = []
    docs.append(local_document)
    return docs

for doc in documents():
    lowers = doc.lower()
    no_punctuation = lowers.translate(punctuation)
    token_dict[doc] = no_punctuation

# this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize(local_document), stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
