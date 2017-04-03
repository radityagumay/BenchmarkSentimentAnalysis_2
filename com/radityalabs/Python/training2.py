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

print(negfeats)

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

test_sentence = "App is fine but you really need to get better at presenting a change log, don't even know what to look for, what has changed, and how can I then give a fair review. It seems to do what it is supposed to do but it's not completely clear. And it takes a lot of changing of settings to get it right, the way I like it. Have a feeling I have to redo that every time I do an update."
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in loc_vocabulary}

print("example:", test_sentence)
print("label:", classifier.classify(featurized_test_sentence))
