# http://textblob.readthedocs.io/en/dev/classifiers.html
from textblob.classifiers import NaiveBayesClassifier
from nltk.stem.snowball import EnglishStemmer
import _pickle as cPickle
import os
import sys, unicodedata
import pymysql

# variable
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()
cur.execute("SELECT reviewBody FROM google_play_crawler.authors17 where length(reviewBody) > 50")

test = []

for i in cur:
    print(i)
    test.append(i)

# training data [("This app is very good", "pos")]
def local_model_data():
    with open(path + "/Python/training_data/training1.pickle", "rb") as handle:
        return cPickle.load(handle)

train = local_model_data()

sentence = "the app was horrible."
cl = NaiveBayesClassifier(train)
prob_dist = cl.prob_classify(sentence)
print(prob_dist.max())
print("probability positive:", round(prob_dist.prob("pos"), 2))
print("probability negative:", round(prob_dist.prob("neg"), 2))
print("accuracy:", cl.accuracy(test))
cl.show_informative_features(5)

# print("classifiy", cl.classify(sentence))
# with open('train.json', 'r') as fp:
#     cl = NaiveBayesClassifier(fp, format="json")
