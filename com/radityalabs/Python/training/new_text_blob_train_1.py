# this use play store twitter corpus
# http://textblob.readthedocs.io/en/dev/classifiers.html#evaluating-classifiers
# http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/
from sklearn.externals import joblib
from textblob import TextBlob
from nltk.metrics import *
from nltk.collocations import BigramCollocationFinder
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
import itertools
import collections
import random
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

def train():
    with open(path + "/Python/bimbingan_data/twitter_train_23536_1.pickle", "rb") as handle:
        return cPickle.load(handle)

def test():
    with open(path + "/Python/bimbingan_data/twitter_test_15691_1.pickle", "rb") as handle:
        return cPickle.load(handle)

def precision_recall(classifier):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    print('pos precision:', classifier.metrics.precision(refsets['pos'], testsets['pos']))
    print('pos recall:', classifier.metrics.recall(refsets['pos'], testsets['pos']))
    print('neg precision:', classifier.metrics.precision(refsets['neg'], testsets['neg']))
    print('neg recall:', classifier.metrics.recall(refsets['neg'], testsets['neg']))

def testing(sentence):
    #collection = train() + test()
    #random.shuffle(collection)
    #testing_this_round = collection[:int((len(collection) * 40) / 100)]
    #training_this_round = collection[int((len(collection) * 60) / 100):]

    classifier = NaiveBayesClassifier(train(), feature_extractor=end_word_extractor)
    blob = TextBlob(sentence, classifier=classifier)
    print(sentence + " label : ", blob.classify())
    print("polarity", blob.sentiment.polarity)  # polarity and subjectivity
    print("subjectivity", blob.sentiment.subjectivity)

    ## calc neg and pos
    sentiment = TextBlob(sentence, classifier=classifier, analyzer=NaiveBayesAnalyzer())
    print("positive", sentiment.sentiment.p_pos)
    print("negative", sentiment.sentiment.p_neg)
    print("Accuracy: {0}".format(classifier.accuracy(test())))

testing(sentence = "Twitter Its descent Im on it a lot more now but 1 thing it needs is a way to chat live with people. I think it would be cool to enhance the bubble tweets so people can live chat on Twitter with the followers.")
