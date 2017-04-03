from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from multiprocessing import Pool
import progressbar
import _pickle as cPickle
import sys, unicodedata
import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()

cur.execute("select reviewBody, label FROM sentiment_analysis.review_label_benchmark_with_polarity where label = 'pos' or label = 'neg'")

stemmer = EnglishStemmer()

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

def is_string_not_empty(string):
    if string == "":
        return False
    return True

def preprocessing(dirty_word):
    lower = dirty_word.lower()          # toLower().Case
    if len(lower) > 3:                  # only len > 3
        stem = stemmer.stem(lower)      # root word
        punc = stem.translate(tbl)      # remove ? ! @ etc
        if is_string_not_empty(punc):   # check if not empty
            stop = punc not in set(stopwords.words('english'))
            if stop:                    # only true we append
                return dirty_word
    else:
        return None

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

db_count = 52150

async def calc_both_neg_and_pos():
    print("\n Initialize local sentiment review")
    initbar = progressbar.ProgressBar(maxval=db_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    initbar.start()
    index = 0
    for i in cur:
        index += 1
        initbar.update(index)
        label = i[1]
        if label == 'neg':
            negfeats.append((word_feats(word_tokenize(i[0])), 'neg'))
        else:
            posfeats.append((word_feats(word_tokenize(i[0])), 'pos'))
    initbar.finish()

def calc_both_neg_and_pos_callback():
    print("worker local sentiment review is done")

## do asyncTask ##
p1 = Pool(processes = 1)
r1 = p1.apply_async(calc_both_neg_and_pos, callback=calc_both_neg_and_pos_callback)
r1.wait()
## done ##

## save negfeats and posfeats to file
async def save_negfeats_and_posfeats():
    with open("negfeats.pickle", "wb") as handle:
        cPickle.dump(negfeats, handle)
    with open("posfeats.pickle", "wb") as handle:
        cPickle.dump(posfeats, handle)

def save_negfeats_and_posfeats_callback():
    print("worker negfeats and posfeats are done")

## do asyncTask ##
p2 = Pool(processes = 1)
r2 = p2.apply_async(save_negfeats_and_posfeats, callback=save_negfeats_and_posfeats_callback)
r2.wait()
## done ##

vocabulary = {}

async def running_sentiment_review_negative():
    print("Initialize sentiment review negative")
    negbar = progressbar.ProgressBar(maxval=len(negids), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    negbar.start()
    index = 0
    for i in negids:
        index += 1
        negbar.update(index)
        for j in movie_reviews.words(fileids=[i]):
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary[j] = j
    negbar.finish()
    print("\n Done added negative movie vocabulary : ", len(vocabulary))
    print("\n")

def running_sentiment_review_negative_callback():
    print("worker sentiment review negative is done")

## do asyncTask ##
p3 = Pool(processes = 1)
r3 = p3.apply_async(running_sentiment_review_negative, callback=running_sentiment_review_negative_callback)
r3.wait()
## done ##

async def running_sentiment_review_positive():
    print("Initialize sentiment review positive")
    posbar = progressbar.ProgressBar(maxval=len(posids), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    posbar.start()
    index = 0
    for i in posids:
        index += 1
        posbar.update(index)
        for j in movie_reviews.words(fileids=[i]):
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary[j] = j
    posbar.finish()
    print("\n Done added positive movie vocabulary : ", len(vocabulary))
    print("\n")

def running_sentiment_review_positive_callback():
    print("worker sentiment review positive is done")

## do asyncTask ##
p4 = Pool(processes = 1)
r4 = p4.apply_async(running_sentiment_review_positive, callback=running_sentiment_review_positive_callback)
r4.wait()
## done ##

async def running_db_sentiment():
    print("Initialize sentiment review database")
    dbbar = progressbar.ProgressBar(maxval=db_count, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    dbbar.start()
    index = 0
    for i in cur:
        dbbar.update(index)
        index += 1
        words = word_tokenize(i[0])
        for j in words:
            is_valid_vocab = preprocessing(j)
            if is_string_not_empty(is_valid_vocab):
                if j not in vocabulary:
                    vocabulary[j] = j
    dbbar.finish()
    print("\n Done added database movie vocabulary : ", len(vocabulary))
    print("\n")

def running_db_sentiment_callback():
    print("worker database is done")

## do asyncTask ##
p5 = Pool(processes = 1)
r5 = p5.apply_async(running_db_sentiment, callback=running_db_sentiment_callback)
r5.wait()
## done ##

cur.close()
conn.close()

async def save_vocabulary_pickle():
    with open("vocabulary.pickle", "wb") as handle:
        cPickle.dump(vocabulary, handle)

def save_vocabulary_pickle_callback():
    print("worker vocabulary pickle is done")

## do asyncTask ##
p6 = Pool(processes = 1)
r6 = p6.apply_async(save_vocabulary_pickle, callback=save_vocabulary_pickle_callback)
r6.wait()
## done ##


# def local_vocabulary():
#     with open('vocabulary.pickle', 'rb') as handle:
#         return cPickle.load(handle)
#
# loc_vocabulary = local_vocabulary()
#
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
# test_sentence = "I love it. It behaves exactly like youd expect. Plus I love the inter-app communication (the fact that I can click a follow button in the gmail app and be taken straight into the twitter app). Good job guys."
# featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in loc_vocabulary}
#
# print("example:", test_sentence)
# print("label:", classifier.classify(featurized_test_sentence))
