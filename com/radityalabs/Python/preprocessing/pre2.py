from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from itertools import chain
from nltk.tokenize import word_tokenize
import asyncio
import _pickle as cPickle
import sys, unicodedata
import os

# variable
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

def is_string_not_empty(string):
    if string == "":
        return False
    return True

def preprocessing(dirty_sentence):
    sentence = ""
    for i in dirty_sentence:
        lower = i.lower()                   # toLower().Case
        if len(lower) > 3:                  # only len > 3
            stem = stemmer.stem(lower)      # root word
            punc = stem.translate(tbl)      # remove ? ! @ etc
            if is_string_not_empty(punc):       # check if not empty
                stop = punc not in set(stopwords.words('english'))
                if stop:                    # only true we append
                    sentence += str(punc) + " "
    return sentence

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

@asyncio.coroutine
def save_negfeats():
    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    with open(path + "/Python/negfeats/negfeats2.pickle", "wb") as handle:
        cPickle.dump(negfeats, handle)

asyncio.get_event_loop().run_until_complete(save_negfeats())

@asyncio.coroutine
def save_posfeats():
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
    with open(path + "/Python/posfeats/posfeats2.pickle", "wb") as handle:
        cPickle.dump(posfeats, handle)

asyncio.get_event_loop().run_until_complete(save_posfeats())

@asyncio.coroutine
def build_vocabulary():
    vocabulary = []
    for n in negids:
        sentence = movie_reviews.raw(fileids=[n])
        vocabulary.append((sentence, 'neg'))
    for p in posids:
        sentence = movie_reviews.raw(fileids=[p])
        vocabulary.append((sentence, 'pos'))
    return vocabulary

@asyncio.coroutine
def save_vocabulary():
    vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in build_vocabulary()]))
    with open(path + "/Python/vocabulary/vocabulary2.pickle", "wb") as handle:
        cPickle.dump(vocabulary, handle)

asyncio.get_event_loop().run_until_complete(save_vocabulary())