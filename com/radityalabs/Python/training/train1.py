# this use play store twitter corpus
# http://textblob.readthedocs.io/en/dev/classifiers.html#evaluating-classifiers
# http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/
from sklearn.externals import joblib
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *
import random
import collections
import itertools
import _pickle as cPickle
import os

# variables
stopset = set(stopwords.words('english')) - {'over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'}
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
    #cl = joblib.load(path + '/Python/bimbingan_data/sklearn-joblib-train-twitter-1.pkl')
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

    test_result = []
    gold_result = []

    for i in range(len(test())):
        test_result.append(classifier.classify(test()[i][0]))
        gold_result.append(test()[i][1])

    print('Clasification report:\n', classification_report(gold_result, test_result))
    print('Confussion matrix:\n', confusion_matrix(gold_result, test_result))

testing(sentence = "Twitter Its descent Im on it a lot more now but 1 thing it needs is a way to chat live with people. I think it would be cool to enhance the bubble tweets so people can live chat on Twitter with the followers.")