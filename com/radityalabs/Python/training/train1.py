# this use play store twitter corpus
# http://textblob.readthedocs.io/en/dev/classifiers.html#evaluating-classifiers
# http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/
from sklearn.externals import joblib
import collections
import _pickle as cPickle
import os

# variables
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats

def load_test():
    with open(path + "/Python/bimbingan_data/train_twitter_corpus_2317.pickle", "rb") as handle:
        return cPickle.load(handle)

def precision_recall(classifier):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    print('pos precision:', classifier.metrics.precision(refsets['pos'], testsets['pos']))
    print('pos recall:', classifier.metrics.recall(refsets['pos'], testsets['pos']))
    print('neg precision:', classifier.metrics.precision(refsets['neg'], testsets['neg']))
    print('neg recall:', classifier.metrics.recall(refsets['neg'], testsets['neg']))

def testing(sentence):
    cl = joblib.load(path + '/Python/bimbingan_data/sklearn-joblib-train.pkl')
    print(cl.classify(sentence))
    print("Accuracy: {0}".format(cl.accuracy(load_test())))
    cl.show_informative_features(5)

testing(sentence = "this app is very good")