# http://streamhacker.com/2010/05/17/text-classification-sentiment-analysis-precision-recall/
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from itertools import chain
import _pickle as cPickle
import os
import progressbar

path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

#trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
#testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
#print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

def build_train():
    vocabulary = []
    firstIndex = 0
    progressBar = progressbar.ProgressBar(maxval=1500, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    progressBar.start()
    for n in negids:
        progressBar.update(firstIndex)
        if firstIndex < 750:
            sentence = movie_reviews.raw(fileids=[n])
            vocabulary.append((sentence, 'neg'))
        else:
            break
        firstIndex += 1
    secondIndex = 0
    index = firstIndex
    for p in posids:
        progressBar.update(index)
        if secondIndex < 750:
            sentence = movie_reviews.raw(fileids=[p])
            vocabulary.append((sentence, 'pos'))
        else:
            break
        secondIndex += 1
        index += 1
    progressBar.finish()
    return vocabulary

def build_test():
    vocabulary = []
    index = 0
    progressBar = progressbar.ProgressBar(maxval=500, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    progressBar.start()

    firstIndex = 0
    for n in negids:
        if index > 750:
            firstIndex += 1
            progressBar.update(firstIndex)
            sentence = movie_reviews.raw(fileids=[n])
            vocabulary.append((sentence, 'neg'))
        index += 1

    index = 0
    for p in posids:
        if index > 750:
            firstIndex += 1
            progressBar.update(firstIndex)
            sentence = movie_reviews.raw(fileids=[p])
            vocabulary.append((sentence, 'pos'))
        index += 1
    progressBar.finish()
    return vocabulary

train_data = build_train()
test_data = build_test()

# train set feature
vocabulary_train = set(chain(*[word_tokenize(i[0].lower()) for i in train_data]))
train_set_feature = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary_train}, tag) for sentence, tag in train_data]

# test set feature
vocabulary_test = set(chain(*[word_tokenize(i[0].lower()) for i in test_data]))
test_set_feature = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary_test}, tag) for sentence, tag in test_data]

def save_train():
    with open(path + "/Python/nltktrain/train.pickle", "wb") as handle:
        cPickle.dump(train_set_feature, handle)

def save_test():
    with open(path + "/Python/nltktrain/test.pickle", "wb") as handle:
        cPickle.dump(test_set_feature, handle)

classifier = NaiveBayesClassifier.train(train_set_feature)
print('accuracy:', nltk.classify.util.accuracy(classifier, test_set_feature))
classifier.show_most_informative_features()
