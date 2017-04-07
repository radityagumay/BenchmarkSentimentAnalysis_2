from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import _pickle as cPickle
import sys, unicodedata
import progressbar
import asyncio
import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()

cur.execute("SELECT reviewBody, label FROM sentiment_analysis.review_label_benchmark_with_polarity where label = 'neg' or label = 'pos' and length(reviewBody) > 50 limit 0, 31828")

stemmer = EnglishStemmer()
db_count = 31828

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

# def is_string_not_empty(string):
#     if string == "":
#         return False
#     return True
#
# def preprocessing(dirty_word):
#     lower = dirty_word.lower()          # toLower().Case
#     if len(lower) > 3:                  # only len > 3
#         stem = stemmer.stem(lower)      # root word
#         punc = stem.translate(tbl)      # remove ? ! @ etc
#         if is_string_not_empty(punc):   # check if not empty
#             stop = punc not in set(stopwords.words('english'))
#             if stop:                    # only true we append
#                 return dirty_word
#     else:
#         return None
#
# def word_feats(words):
#     return dict([(word, True) for word in words])
#
# negids = movie_reviews.fileids('neg')
# posids = movie_reviews.fileids('pos')
#
# negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
# posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

# @asyncio.coroutine
# def load_negative():
#     with open("corpus/polarity-data-negative.txt", "r") as file:
#         data = file.read()
#         negfeats.append((word_feats(word_tokenize(data)), 'neg'))
#
# asyncio.get_event_loop().run_until_complete(load_negative())
#
# @asyncio.coroutine
# def load_positive():
#     with open("corpus/polarity-data-positive.txt", "r") as file:
#         data = file.read()
#         posfeats.append((word_feats(word_tokenize(data)), 'pos'))
#
# asyncio.get_event_loop().run_until_complete(load_positive())

def close_connection():
    cur.close()
    conn.close()

@asyncio.coroutine
def running_db_sentiment():
    negative = ""
    positive = ""
    dbbar = progressbar.ProgressBar(maxval=db_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    dbbar.start()
    index = 0
    for i in cur:
        dbbar.update(index)
        index += 1
        label = i[1]
        if label == 'neg':
            negative += i[0] + " . "
        else:
            positive += i[0] + " . "
    print(negative)
    #negfeats.append((word_feats(word_tokenize(negative)), 'neg'))
    #posfeats.append((word_feats(word_tokenize(positive)), 'pos'))
    dbbar.finish()
    close_connection()

asyncio.get_event_loop().run_until_complete(running_db_sentiment())

# @asyncio.coroutine
# def save_negfeats():
#     with open("negfeats6.pickle", "wb") as handle:
#         cPickle.dump(negfeats, handle)
#
# @asyncio.coroutine
# def save_posfeats():
#     with open("posfeats6.pickle", "wb") as handle:
#         cPickle.dump(posfeats, handle)
#
# asyncio.get_event_loop().run_until_complete(save_negfeats())
# asyncio.get_event_loop().run_until_complete(save_posfeats())

# @asyncio.coroutine
# def load_negfeats():
#     with open("negfeats4.pickle", "rb") as handle:
#         return cPickle.load(handle)
#
# neg_coroutine = asyncio.get_event_loop()
# load_negfeats = neg_coroutine.run_until_complete(load_negfeats())
# print(load_negfeats)
#
# @asyncio.coroutine
# def load_posfeats():
#     with open("posfeats4.pickle", "rb") as handle:
#         return cPickle.load(handle)
#
# pos_coroutine = asyncio.get_event_loop()
# load_posfeats = pos_coroutine.run_until_complete(load_negfeats())
# print(load_posfeats)