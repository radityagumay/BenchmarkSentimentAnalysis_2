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

# functions
def word_feats(words):
    return dict([(word, True) for word in words])

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

def close_connection():
    cur.close()
    conn.close()

full_sentence = []

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
        clean = preprocessing(word_tokenize(i[0]))
        if label == 'neg':
            full_sentence.append((clean, label))
        else:
            full_sentence.append((clean, label))
    #print(vocabulary)
    #negfeats.append((word_feats(word_tokenize(negative)), 'neg'))
    #posfeats.append((word_feats(word_tokenize(positive)), 'pos'))
    close_connection()

asyncio.get_event_loop().run_until_complete(running_db_sentiment())

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in full_sentence]))
print(vocabulary)

# @asyncio.coroutine
# def save_negfeats():
#     with open(path + "/Python/negfeats/negfeats1.pickle", "wb") as handle:
#         cPickle.dump(negfeats, handle)
#         print("Saving pickle negfeats is done")
#
# asyncio.get_event_loop().run_until_complete(save_negfeats())
#
# @asyncio.coroutine
# def save_posfeats():
#     with open(path + "/Python/posfeats/posfeats1.pickle", "wb") as handle:
#         cPickle.dump(posfeats, handle)
#         print("Saving pickle posfeats is done")
#
# asyncio.get_event_loop().run_until_complete(save_posfeats())
#
# @asyncio.coroutine
# def save_vocabulary_pickle():
#     with open(path + "/Python/vocabulary/vocabulary1.pickle", "wb") as handle:
#         cPickle.dump(vocabulary, handle)
#         print("Saving pickle vocabulary is done")
#
# asyncio.get_event_loop().run_until_complete(save_vocabulary_pickle())