# this use play store twitter corpus
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
training()