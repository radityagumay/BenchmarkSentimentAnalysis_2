from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from itertools import chain
import sys, unicodedata
import _pickle as cPickle
import pymysql
import os

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()
cur.execute("SELECT reviewBody,label FROM review_label_benchmark_with_polarity")
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

# when we load data from database, we should consifer use below approach to make
# data preprocessing
# 1. load data from database
# 2. tokenized
# 3. Stemming
# 4. punctuation
# 5. stopword

def is_string_empty(string):
    if string == "":
        return False
    return True

def preprocessing(dirty_sentence):
    sentence = ""
    for i in dirty_sentence:
        lower = i.lower()  # toLower().Case
        if len(lower) > 3:  # only len > 3
            stem = stemmer.stem(lower)  # root word
            punc = stem.translate(tbl)  # remove ? ! @ etc
            if is_string_empty(punc):  # check if not empty
                stop = punc not in set(stopwords.words('english'))
                if stop:  # only true we append
                    sentence += str(punc) + " "
    return sentence

training_data = []

def build_train():
    for r in cur:
        clean = preprocessing(word_tokenize(r[0]))
        training_data.append((clean, r[1]))

build_train()
cur.close()
conn.close()
vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))

print(vocabulary)

# write model vocabulary, for usage later
def write_model_vocabulary():
    with open(path + "/Python/vocabulary/vocabulary.pickle", "wb") as handle:
        cPickle.dump(vocabulary, handle)
        print("done write model vocabulary")

# create feature set from our data training
# feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag)
#                for sentence, tag in training_data]
#
# def write_feature():
#     with open("features.pickle", "wb") as handle:
#         cPickle.dump(feature_set, handle)
#         print("done write feature")
#
# write_feature()
