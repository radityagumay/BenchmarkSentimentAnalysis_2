# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
from textblob.classifiers import NaiveBayesClassifier
from sklearn.externals import joblib
from textblob import TextBlob
import _pickle as cPickle
import os
import pymysql
import asyncio

# variables
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
items_total = 39227
items_train_count = 23536
items_test_count = 15691

def connection():
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root', passwd='',
        db='sentiment_analysis')
    cursor = conn.cursor()
    return cursor, conn

def close_connection(cursor, conn):
    conn.close()
    cursor.close

def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats

def sample_sentence():
    return "This app is never good enough"

def train_and_test(train, test):
    cl = NaiveBayesClassifier(train, feature_extractor=end_word_extractor)
    blob = TextBlob(sample_sentence(), classifier=cl)
    print(blob.classify())
    print("Accuracy: {0}".format(cl.accuracy(test)))

def run_me():
    cursor, conn = connection()
    cursor.execute("SELECT reviewBody, label FROM sentiment_analysis.review_label_benchmark_with_polarity where length(reviewBody) > 30 and (label = 'pos' or label = 'neg') limit 0, 39227")
    datas = []
    for data in cursor:
        datas.append((data[0], data[1]))
    train = datas[:23536]
    test = datas[23536:]
    train_and_test(train, test)
    close_connection(cursor, conn)

run_me()

# train
# @asyncio.coroutine
# def load_train():
#     train = []
#     for item in cursor:
#         train.append((item[0], item[1]))
#     return train
# train = asyncio.get_event_loop().run_until_complete(load_train())
# print(train)
#
# # test
# cursor.execute("SELECT reviewBody, label FROM google_play_crawler.authors17 where length(reviewBody) > 30 and (label = 'pos' or label = 'neg') limit 0, 2317")
# @asyncio.coroutine
# def load_test():
#     test = []
#     for item in cursor:
#         test.append((item[0], item[1]))
#     return test
# test = asyncio.get_event_loop().run_until_complete(load_test())
# print(test)
#
# @asyncio.coroutine
# def save_train():
#     with open(path + "/Python/bimbingan_data/train_twitter_corpus_34636_1.pickle", "wb") as handle:
#         cPickle.dump(train, handle)
#         print("Saving train is done")
#
# @asyncio.coroutine
# def save_test():
#     with open(path + "/Python/bimbingan_data/train_twitter_corpus_2317_1.pickle", "wb") as handle:
#         cPickle.dump(test, handle)
#         print("Saving test is done")
#
# asyncio.get_event_loop().run_until_complete(save_train())
# asyncio.get_event_loop().run_until_complete(save_test())
#
# cursor.close()
# connection.close()
#
# def end_word_extractor(document):
#     tokens = document.split()
#     first_word, last_word = tokens[0], tokens[-1]
#     feats = {}
#     feats["first({0})".format(first_word)] = True
#     feats["last({0})".format(last_word)] = False
#     return feats
#
# cl = NaiveBayesClassifier(train, feature_extractor=end_word_extractor)
# joblib.dump(cl, path + '/Python/bimbingan_data/sklearn-joblib-train-twitter-1.pkl')