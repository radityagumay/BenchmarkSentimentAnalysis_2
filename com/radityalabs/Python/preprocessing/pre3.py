# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
from textblob.classifiers import NaiveBayesClassifier
from sklearn.externals import joblib
from nltk.corpus import movie_reviews
from sklearn.model_selection import KFold
import _pickle as cPickle
import random
import os

# variables
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

def load_train():
    with open(path + "/Python/bimbingan_data/train_twitter_corpus_34636_1.pickle", "rb") as handle:
        return cPickle.load(handle)

def load_test():
    with open(path + "/Python/bimbingan_data/train_twitter_corpus_2317_1.pickle", "rb") as handle:
        return cPickle.load(handle)

def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats

def word_feats(words):
    return dict([(word, True) for word in words])

def save_train(train):
    with open(path + "/Python/bimbingan_data/train_twitter_corpus_34636_nltk_3.pickle", "wb") as handle:
        cPickle.dump(train, handle)
        print("Saving train is done")

def save_test(test):
    with open(path + "/Python/bimbingan_data/train_twitter_corpus_2317_nltk_3.pickle", "wb") as handle:
        cPickle.dump(test, handle)
        print("Saving test is done")

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

    collect_train = load_train() + train_data
    save_train(collect_train)
    save_test(load_test() + test_data)

    cl = NaiveBayesClassifier(collect_train, feature_extractor=end_word_extractor)
    joblib.dump(cl, path + '/Python/bimbingan_data/sklearn-joblib-train-combine-twitter-nltk-3.pkl')

run()
