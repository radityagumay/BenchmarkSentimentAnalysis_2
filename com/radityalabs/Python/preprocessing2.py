import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import progressbar
import _pickle as cPickle
import sys, unicodedata
import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()

cur.execute("SELECT reviewBody,label FROM review_label_benchmark_with_polarity")

stemmer = EnglishStemmer()

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))


def is_string_not_empty(string):
    if string == "":
        return False
    return True

def preprocessing(dirty_word):
    lower = dirty_word.lower()  # toLower().Case
    if len(lower) > 3:  # only len > 3
        stem = stemmer.stem(lower)  # root word
        punc = stem.translate(tbl)  # remove ? ! @ etc
        if is_string_not_empty(punc):  # check if not empty
            stop = punc not in set(stopwords.words('english'))
            if stop:  # only true we append
                return dirty_word
    else:
        return None

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

vocabulary = {}

negbar = progressbar.ProgressBar(maxval=len(negids), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
negbar.start()
index = 0
for i in negids:
    negbar.update(index + 1)
    for j in movie_reviews.words(fileids=[i]):
        is_valid_vocab = preprocessing(j)
        if is_string_not_empty(is_valid_vocab):
            if j not in vocabulary:
                vocabulary[j] = j
    index += 1

print("\n")
print("Done added negative movie vocabulary : ", len(vocabulary))

posbar = progressbar.ProgressBar(maxval=len(posids), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
posbar.start()
index = 0
for i in posids:
    posbar.update(index + 1)
    for j in movie_reviews.words(fileids=[i]):
        is_valid_vocab = preprocessing(j)
        if is_string_not_empty(is_valid_vocab):
            if j not in vocabulary:
                vocabulary[j] = j
    index += 1

print("\n")
print("Done added positive movie vocabulary : ", len(vocabulary))

dbbar = progressbar.ProgressBar(maxval=len(cur), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
dbbar.start()
index = 0
for i in cur:
    dbbar.update(index + 1)
    words = word_tokenize(i[0])
    for j in words:
        is_valid_vocab = preprocessing(j)
        if is_string_not_empty(is_valid_vocab):
            if j not in vocabulary:
                vocabulary[j] = j
    index += 1

print("\n")
print("Done added database movie vocabulary : ", len(vocabulary))

cur.close()
conn.close()

with open("vocabulary.pickle", "wb") as handle:
    cPickle.dump(vocabulary, handle)

print(vocabulary)


# negcutoff = len(negfeats) * 3 / 4
# poscutoff = len(posfeats) * 3 / 4
#
# trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
# testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
# print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))
#
# classifier = NaiveBayesClassifier.train(trainfeats)
# print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
# classifier.show_most_informative_features()
#
# test_sentence = "This best movie ever!!"
# featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in vocabulary}
#
# print("example:", test_sentence)
# print("label:", classifier.classify(featurized_test_sentence))
