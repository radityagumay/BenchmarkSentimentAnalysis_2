# http://stackabuse.com/python-async-await-tutorial/
# this use nltk corpus
# this code using our corpus
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import nltk.classify.util
import sys, unicodedata
import _pickle as cPickle
import os

# variable
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

# functions
def local_vocabulary():
    with open(path + "/Python/vocabulary/vocabulary2.pickle", "rb") as handle:
        return cPickle.load(handle)

def local_negfeats():
    with open(path + "/Python/negfeats/negfeats2.pickle", "rb") as handle:
        return cPickle.load(handle)

def local_posfeats():
    with open(path + "/Python/posfeats/posfeats2.pickle", "rb") as handle:
        return cPickle.load(handle)

def local_model_data():
    with open(path + "/Python/model/model1.pickle", "rb") as handle:
        return cPickle.load(handle)

vocabulary = local_vocabulary()
negfeats = local_negfeats()
posfeats = local_posfeats()
model = local_model_data()

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]

print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

def training():
    for i in model:
        test_sentence = i[0]
        featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in vocabulary}
        print("\n")
        print("==================================")
        print("sentence :", test_sentence)
        print("label :", classifier.classify(featurized_test_sentence))
        print("expected label :", i[1])
        print("==================================")
#training()

test_sentence = "The new update makes twitter taking soooo long to load my timeline and mentions. I have to wait for approximately 30 minutes without closing it before it finally loads. Before twitter was one of the lightest app of social media and now it becomes heavy taking around 80% of my RAM. The new automatically saving picture is also not helping we need an option to turn the feature off."
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in vocabulary}
print("\n")
print("==================================")
print("sentence :", test_sentence)
print("label :", classifier.classify(featurized_test_sentence))
print("expected label :", "pos")
print("==================================")
