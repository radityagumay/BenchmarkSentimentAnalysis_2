# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
from textblob.classifiers import NaiveBayesClassifier
from sklearn.externals import joblib
from textblob import TextBlob
from nltk.corpus import movie_reviews
import _pickle as cPickle
import random
import os

# variables
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

# instantiate
def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats

def save_train(train):
    with open(path + "/Python/bimbingan_data/nltk_train_1500_2.pickle", "wb") as handle:
        cPickle.dump(train, handle)
        print("saving train data's is done")

def save_test(test):
    with open(path + "/Python/bimbingan_data/nltk_test_500_2.pickle", "wb") as handle:
        cPickle.dump(test, handle)
        print("saving test data's is done")

def sentence():
    return "This app is never good enough"

def run():
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
    collection = []
    for i in negids:
        collection.append((movie_reviews.raw(fileids=[i]), 'neg'))
    for i in posids:
        collection.append((movie_reviews.raw(fileids=[i]), 'pos'))
    random.shuffle(collection)
    train_data = collection[:1500]
    test_data = collection[1500:]
    save_train(train_data)
    save_test(test_data)

    cl = NaiveBayesClassifier(train_data, feature_extractor=end_word_extractor)
    blob = TextBlob(sentence(), classifier=cl)
    print(blob.classify())
    print("Accuracy: {0}".format(cl.accuracy(test_data)))
    #cl = NaiveBayesClassifier(train_data)
    #joblib.dump(cl, path + '/Python/bimbingan_data/sklearn-joblib-train-nltk-2.pkl')

run()