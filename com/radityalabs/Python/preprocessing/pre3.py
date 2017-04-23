# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
from nltk.corpus import movie_reviews
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import pymysql
import _pickle as cPickle
import random
import os

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

def query(cursor):
    return cursor.execute("SELECT reviewBody, label FROM sentiment_analysis.review_label_benchmark_with_polarity where length(reviewBody) > 30 and (label = 'pos' or label = 'neg') limit 0, 39227")

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

def sentence():
    return "This app is never good enough"

def added_from_movie_review_train_and_test():
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
    collection = []
    for i in negids:
        collection.append((movie_reviews.raw(fileids=[i]), 'neg'))
    for i in posids:
        collection.append((movie_reviews.raw(fileids=[i]), 'pos'))
    random.shuffle(collection)
    train_nltk = collection[:1500]
    test_nltk = collection[1500:]
    return train_nltk, test_nltk

def train_and_test(train, test):
    train_nltk, test_nltk = added_from_movie_review_train_and_test()
    new_train = train + train_nltk
    new_test = test + test_nltk

    save_train(new_train)
    save_test(new_test)

    cl = NaiveBayesClassifier(new_train, feature_extractor=end_word_extractor)
    blob = TextBlob(sentence(), classifier=cl)
    print(blob.classify())
    print("Accuracy: {0}".format(cl.accuracy(new_test)))

def save_train(train):
    with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "wb") as handle:
        cPickle.dump(train, handle)
        print("saving train data's is done")

def save_test(test):
    with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "wb") as handle:
        cPickle.dump(test, handle)
        print("saving test data's is done")

def run_me():
    cursor, conn = connection()
    query(cursor)
    datas = []
    for data in cursor:
        datas.append((data[0], data[1]))
    train = datas[:23536]
    test = datas[23536:]
    train_and_test(train, test)
    close_connection(cursor, conn)

run_me()
