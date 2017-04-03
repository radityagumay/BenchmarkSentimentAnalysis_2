import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import _pickle as cPickle

def local_vocabulary():
    with open('vocabulary.pickle', 'rb') as handle:
        return cPickle.load(handle)

loc_vocabulary = local_vocabulary()

def local_negfeats():
    with open('negfeats.pickle', 'rb') as handle:
        return cPickle.load(handle)

def local_posfeats():
    with open('posfeats.pickle', 'rb') as handle:
        return cPickle.load(handle)

negfeats = local_negfeats()
posfeats = local_posfeats()

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

test_sentence = "this best app ever"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in loc_vocabulary}

print("example:", test_sentence)
print("label:", classifier.classify(featurized_test_sentence))
