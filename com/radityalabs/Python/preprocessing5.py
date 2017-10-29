# http://stackabuse.com/python-async-await-tutorial/
# this code combine both nltk and my own corpus
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

from nltk.corpus import stopwords
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

cur = conn.cursor()
cur.execute("SELECT reviewBody, label FROM sentiment_analysis.review_label_benchmark_with_polarity where label = 'neg' or label = 'pos' and length(reviewBody) > 50 limit 0, 31828")
stemmer = EnglishStemmer()
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

def is_string_not_empty(string):
    if string == "":
        return False
    return True

def preprocessing(dirty_word):
    lower = dirty_word.lower()          # toLower().Case
    if len(lower) > 3:                  # only len > 3
        stem = stemmer.stem(lower)      # root word
        punc = stem.translate(tbl)      # remove ? ! @ etc
        if is_string_not_empty(punc):   # check if not empty
            stop = punc not in set(stopwords.words('english'))
            if stop:                    # only true we append
                return dirty_word
    else:
        return None

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

db_count = 31828
vocabulary = {}

def close_connection():
    cur.close()
    conn.close()

@asyncio.coroutine
def running_db_sentiment():
    print("Initialize sentiment review database")
    dbbar = progressbar.ProgressBar(maxval=2000, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    dbbar.start()
    index = 0
    # for i in cur:
    #     dbbar.update(index)
    #     index += 1
    #     words = word_tokenize(i[0])
    #     for j in words:
    #         is_valid_vocab = preprocessing(j)
    #         if is_string_not_empty(is_valid_vocab):
    #             if j not in vocabulary:
    #                 vocabulary[j] = j
    for i in negids:
        dbbar.update(index)
        index += 1
        for j in movie_reviews.words(fileids=[i]):
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary[j] = j
    for i in posids:
        dbbar.update(index)
        index += 1
        for j in movie_reviews.words(fileids=[i]):
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary[j] = j
    dbbar.finish()
    #close_connection()

asyncio.get_event_loop().run_until_complete(running_db_sentiment())

@asyncio.coroutine
def save_vocabulary_pickle():
    with open(path + "/Python/vocabulary/vocabulary5.pickle", "wb") as handle:
        cPickle.dump(vocabulary, handle)
        print("Saving pickle vocabulary is done")

asyncio.get_event_loop().run_until_complete(save_vocabulary_pickle())