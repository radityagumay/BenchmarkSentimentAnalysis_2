# https://gist.githubusercontent.com/anabranch/48c5c0124ba4e162b2e3/raw/6b1acf8391b15ad3a663beb7e685e6835c964036/tfpdf.py
# http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
from __future__ import division
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import _pickle as cPickle
import sys, unicodedata, os
import math

path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

class Similarity:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tablePunctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    def DEBUG(self, boolean):
        self.is_debug = boolean

    def log(self, func, message):
        if self.is_debug:
            print("=========== " + func.capitalize() + " ===========")
            print(message)
            print("")

    def preprocessing_documents(self, documents):
        docs = []
        for document in documents:
            sentence = ""
            tokens = word_tokenize(document[0])
            for token in tokens:
                if len(token) > 3:
                    stem = self.stemmer.stem(token)
                    punct = stem.translate(self.tablePunctuation)
                    if punct is not None:
                        stop = punct not in set(stopwords.words('english'))
                        if stop:
                            sentence += str(punct.lower()) + " "
            docs.append((sentence, document[1]))
        return docs

    def save_documents(self):
        train = []
        test = []
        with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "rb") as handle:
            train = cPickle.load(handle)
        with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "rb") as handle:
            test = cPickle.load(handle)
        documents = self.preprocessing_documents(train + test)

        self.log("save_documents", documents)

        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "wb") as handle:
            cPickle.dump(test, handle)

similary = Similarity()
similary.DEBUG(True)
similary.save_documents()
