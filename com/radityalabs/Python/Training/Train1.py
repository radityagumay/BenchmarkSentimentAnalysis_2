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

def local_training_data():
    with open(path + "/Python/training_data/training1.pickle", "rb") as handle:
        return cPickle.load(handle)

vocabulary = local_vocabulary()
training_data = local_training_data()
negfeats = local_negfeats()
posfeats = local_posfeats()

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in
               training_data]
classifier = NaiveBayesClassifier.train(feature_set)
# classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

test_sentence = "This very bad app ever"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in vocabulary}

print("example:", test_sentence)
print("label:", classifier.classify(featurized_test_sentence))
