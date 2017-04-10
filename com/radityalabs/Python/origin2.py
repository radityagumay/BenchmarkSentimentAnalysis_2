# http://stackabuse.com/python-async-await-tutorial/
# this code using our corpus
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import nltk.classify.util
import sys, unicodedata

stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

def is_string_not_empty(string):
    if string == "":
        return False
    return True

def preprocessing(dirty_sentence):
    sentence = ""
    for i in dirty_sentence:
        lower = i.lower()                   # toLower().Case
        if len(lower) > 3:                  # only len > 3
            stem = stemmer.stem(lower)      # root word
            punc = stem.translate(tbl)      # remove ? ! @ etc
            if is_string_empty(punc):       # check if not empty
                stop = punc not in set(stopwords.words('english'))
                if stop:                    # only true we append
                    sentence += str(punc) + " "
    return sentence

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

def build_vocabulary():
    vocabulary = []
    for n in negids:
        for j in movie_reviews.words(fileids=[n]):
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary.append((j, 'neg'))
    for p in posids:
        for j in movie_reviews.words(fileids=[p]):
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary.append((j, 'pos'))
    return vocabulary

vocabulary = build_vocabulary()

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

test_sentence = "This app is never good enough"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in vocabulary}

print("example:", test_sentence)
print("label:", classifier.classify(featurized_test_sentence))
