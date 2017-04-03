import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import _pickle as cPickle

def local_vocabulary():
    with open('vocabulary.pickle', 'rb') as handle:
        return cPickle.load(handle)

loc_vocabulary = local_vocabulary()

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

test_sentence = "I love it. It behaves exactly like youd expect. Plus I love the inter-app communication (the fact that I can click a follow button in the gmail app and be taken straight into the twitter app). Good job guys."
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in loc_vocabulary}

print("example:", test_sentence)
print("label:", classifier.classify(featurized_test_sentence))
