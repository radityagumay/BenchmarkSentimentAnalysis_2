# http://textblob.readthedocs.io/en/dev/classifiers.html
from textblob.classifiers import NaiveBayesClassifier
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import _pickle as cPickle
import os
import sys, unicodedata
import pymysql

# variable
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
test = []
train = []

db_google = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='google_play_crawler')

cursor_google = db_google.cursor()
cursor_google.execute("SELECT reviewBody, label FROM google_play_crawler.authors17 limit 2000, 2377")

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

# populate for training data
for data in cursor_google:
    #print("dirty sentence", data[0])
    clean_word = word_tokenize(data[0])
    clean_sentence = ""
    for i in clean_word:
        preprocessing_word = preprocessing(i)
        if preprocessing_word:
            clean_sentence += str(i) + " "
    #print("clean sentence", clean_sentence)
    test.append((clean_sentence, data[1]))

def save_text_blob_train():
    with open("text-blob-test.pickle", "wb") as handle:
        cPickle.dump(train, handle)

save_text_blob_train()

cursor_google.close()
db_google.close()

# def local_model_data():
#     with open(path + "/Python/training_data/training1.pickle", "rb") as handle:
#         return cPickle.load(handle)

#train = local_model_data()

# sentence = "the app was horrible."
# cl = NaiveBayesClassifier(train)
# prob_dist = cl.prob_classify(sentence)
# print(prob_dist.max())
# print("probability positive:", round(prob_dist.prob("pos"), 2))
# print("probability negative:", round(prob_dist.prob("neg"), 2))
# print("accuracy:", cl.accuracy(test))
# cl.show_informative_features(5)

# print("classifiy", cl.classify(sentence))
# with open('train.json', 'r') as fp:
#     cl = NaiveBayesClassifier(fp, format="json")
