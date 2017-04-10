from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from itertools import chain
import asyncio
import progressbar
import _pickle as cPickle
import sys, unicodedata
import pymysql
import os

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

# initialize
cur = conn.cursor()
cur.execute("SELECT reviewBody, label FROM sentiment_analysis.review_label_benchmark_with_polarity where label = 'neg' or label = 'pos' and length(reviewBody) > 50 limit 0, 31828")
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

# variables
db_count = 31828
negfeats = []
posfeats = []
full_sentence = []

# functions
def word_feats(words):
    return dict([(word, True) for word in words])

def close_connection():
    cur.close()
    conn.close()

def word_feats(words):
    return dict([(word, True) for word in words])

@asyncio.coroutine
def running_db_sentiment():
    pBar1 = progressbar.ProgressBar(maxval=db_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    pBar1.start()
    index = 0
    print("populate sentence")
    for i in cur:
        pBar1.update(index)
        index += 1
        label = i[1]
        if label == 'neg':
            negfeats.append((word_feats(word_tokenize(i[0])), 'neg'))
            full_sentence.append((i[0], label))
        else:
            posfeats.append((word_feats(word_tokenize(i[0])), 'pos'))
            full_sentence.append((i[0], label))
    close_connection()

asyncio.get_event_loop().run_until_complete(running_db_sentiment())

@asyncio.coroutine
def save_training_data():
    with open(path + "/Python/training_data/training1.pickle", "wb") as handle:
        cPickle.dump(full_sentence, handle)
        print("Saving training data is done")

asyncio.get_event_loop().run_until_complete(save_training_data())

@asyncio.coroutine
def save_negfeats():
    with open(path + "/Python/negfeats/negfeats1.pickle", "wb") as handle:
        cPickle.dump(negfeats, handle)
        print("Saving pickle negfeats is done")

asyncio.get_event_loop().run_until_complete(save_negfeats())

@asyncio.coroutine
def save_posfeats():
    with open(path + "/Python/posfeats/posfeats1.pickle", "wb") as handle:
        cPickle.dump(posfeats, handle)
        print("Saving pickle posfeats is done")

asyncio.get_event_loop().run_until_complete(save_posfeats())

@asyncio.coroutine
def save_vocabulary_pickle():
    vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in full_sentence]))
    with open(path + "/Python/vocabulary/vocabulary1.pickle", "wb") as handle:
        cPickle.dump(vocabulary, handle)
        print("Saving pickle vocabulary is done")

asyncio.get_event_loop().run_until_complete(save_vocabulary_pickle())