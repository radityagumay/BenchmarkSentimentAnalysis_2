from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *
from nltk.tokenize import word_tokenize
from itertools import chain
# from nltk.metrics import BigramAssocMeasures
import _pickle as cPickle
import itertools
import os
import random
import collections
import nltk.classify.util
import sys, unicodedata

# variables
stopset = set(stopwords.words('english')) - {'over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'}
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

my_pos_data = []
my_neg_data = []
def my_train_data():
    with open(path + "/Python/bimbingan_data/twitter_train_23536_1.pickle", "rb") as handle:
        files = cPickle.load(handle)
        for file in files:
            if file[1] == 'pos':
                my_pos_data.append(file[0])
            elif file[1] == 'neg':
                my_neg_data.append(file[0])

def my_test_data():
    with open(path + "/Python/bimbingan_data/twitter_test_15691_1.pickle", "rb") as handle:
        files = cPickle.load(handle)
        for file in files:
            if file[1] == 'pos':
                my_pos_data.append(file[0])
            elif file[1] == 'neg':
                my_neg_data.append(file[0])

my_train_data()
my_test_data()

def word_split(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new

def word_split_sentiment(data):
    data_new = []
    for (word, sentiment) in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append((word_filter, sentiment))
    return data_new

def word_feats(words):
    return dict([(word, True) for word in words])

def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def bigram_word_feats_stopwords(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in stopset])

def sentence():
    return "Twitter Its descent Im on it a lot more now but 1 thing it needs is a way to chat live with people. I think it would be cool to enhance the bubble tweets so people can live chat on Twitter with the followers."

# Calculating Precision, Recall & F-measure
def evaluate_classifier_default(featx):
    negfeats = [(featx(f), 'neg') for f in word_split(my_neg_data)]
    posfeats = [(featx(f), 'pos') for f in word_split(my_pos_data)]

    negcutoff = len(negfeats) * 3 / 4
    poscutoff = len(posfeats) * 3 / 4

    trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
    testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]

    classifierName = 'Naive Bayes'
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    accuracy = nltk.classify.util.accuracy(classifier, testfeats)
    pos_precision = nltk.precision(refsets['pos'], testsets['pos'])
    pos_recall = nltk.recall(refsets['pos'], testsets['pos'])
    pos_fmeasure = nltk.f_measure(refsets['pos'], testsets['pos'])
    neg_precision = nltk.precision(refsets['neg'], testsets['neg'])
    neg_recall = nltk.recall(refsets['neg'], testsets['neg'])
    neg_fmeasure = nltk.f_measure(refsets['neg'], testsets['neg'])

    print("")
    print('---------------------------------------')
    print('SINGLE FOLD RESULT ' + '(' + classifierName + ')')
    print('---------------------------------------')
    print('sentence:', sentence())
    print('label:', classifier.classify(word_feats(sentence())))
    print('accuracy:', accuracy)
    print('precision', (pos_precision + neg_precision) / 2)
    print('recall', (pos_recall + neg_recall) / 2)
    print('f-measure', (pos_fmeasure + neg_fmeasure) / 2)
    #classifier.show_most_informative_features()
    print("")

def evaluate_classifier(featx):
    negfeats = [(featx(f), 'neg') for f in word_split(my_neg_data)]
    posfeats = [(featx(f), 'pos') for f in word_split(my_pos_data)]

    ## CROSS VALIDATION
    trainfeats = negfeats + posfeats

    # SHUFFLE TRAIN SET
    # As in cross validation, the test chunk might have only negative or only positive data
    random.shuffle(trainfeats)
    n = 5  # 5-fold cross-validation

    subset_size = len(trainfeats) / n
    accuracy = []
    pos_precision = []
    pos_recall = []
    neg_precision = []
    neg_recall = []
    pos_fmeasure = []
    neg_fmeasure = []
    cv_count = 1
    classifier = ""
    for i in range(n):
        testing_this_round = trainfeats[i * int(subset_size):][:int(subset_size)]
        training_this_round = trainfeats[:i * int(subset_size)] + trainfeats[(i + 1) * int(subset_size):]

        classifierName = 'Naive Bayes'
        classifier = NaiveBayesClassifier.train(training_this_round)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats, label) in enumerate(testing_this_round):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

        cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
        cv_pos_precision = nltk.precision(refsets['pos'], testsets['pos'])
        cv_pos_recall = nltk.recall(refsets['pos'], testsets['pos'])
        cv_pos_fmeasure = nltk.f_measure(refsets['pos'], testsets['pos'])
        cv_neg_precision = nltk.precision(refsets['neg'], testsets['neg'])
        cv_neg_recall = nltk.recall(refsets['neg'], testsets['neg'])
        cv_neg_fmeasure = nltk.f_measure(refsets['neg'], testsets['neg'])

        accuracy.append(cv_accuracy)
        pos_precision.append(cv_pos_precision)
        pos_recall.append(cv_pos_recall)
        neg_precision.append(cv_neg_precision)
        neg_recall.append(cv_neg_recall)
        pos_fmeasure.append(cv_pos_fmeasure)
        neg_fmeasure.append(cv_neg_fmeasure)

        cv_count += 1

    print('---------------------------------------')
    print('N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')')
    print('---------------------------------------')
    print('sentence:', sentence())
    print('label:', classifier.classify(bigram_word_feats_stopwords(sentence())))
    print('accuracy:', sum(accuracy) / n)
    print('precision', (sum(pos_precision) / n + sum(neg_precision) / n) / 2)
    print('recall', (sum(pos_recall) / n + sum(neg_recall) / n) / 2)
    print('f-measure', (sum(pos_fmeasure) / n + sum(neg_fmeasure) / n) / 2)
    print("")

evaluate_classifier_default(word_feats)
#evaluate_classifier(stopword_filtered_word_feats)
#evaluate_classifier(bigram_word_feats)
evaluate_classifier(bigram_word_feats_stopwords)