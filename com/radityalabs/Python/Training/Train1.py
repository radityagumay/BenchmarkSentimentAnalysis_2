import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import _pickle as cPickle
import os

# variable
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

# functions
def local_vocabulary():
    with open(path + "/Python/vocabulary/vocabulary1.pickle", "rb") as handle:
        return cPickle.load(handle)


def local_negfeats():
    with open(path + "/Python/negfeats/negfeats1.pickle", "rb") as handle:
        return cPickle.load(handle)

def local_posfeats():
    with open(path + "/Python/posfeats/posfeats1.pickle", "rb") as handle:
        return cPickle.load(handle)

loc_vocabulary = local_vocabulary()
negfeats = local_negfeats()
posfeats = local_posfeats()

print(negfeats)

