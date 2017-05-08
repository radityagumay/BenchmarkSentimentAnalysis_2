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

    def tokenize(self, text):
        collection_tokens = []
        tokens = word_tokenize(text)
        for token in tokens:
            if len(token) > 3:
                stem = self.stemmer.stem(token)
                punct = stem.translate(self.tablePunctuation)
                if punct is not None:
                    stop = punct not in set(stopwords.words('english'))
                    if stop:
                        if punct not in collection_tokens:
                            collection_tokens.append(punct.lower())
        return collection_tokens

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
        with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "rb") as handle:
            train = cPickle.load(handle)
        with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "rb") as handle:
            test = cPickle.load(handle)
        documents = self.preprocessing_documents(train + test)

        #self.log("save_documents", documents)

        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "wb") as handle:
            cPickle.dump(documents, handle)

    def load_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            documents = cPickle.load(handle)
            #self.log("load_documents",  documents)
            return documents

    def sublinear_term_frequency(self, term, tokenized_document):
        count = tokenized_document.count(term)
        if count == 0:
            return 0
        return 1 + math.log(count)

    # save all tokens into vector #
    def save_vector_space(self, words):
        with open(path + "/Python/bimbingan_data/tfidf-final-vector-space-words.pickle", "wb") as handle:
            cPickle.dump(words, handle)

    def load_vector_space(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-vector-space-words.pickle", "rb") as handle:
            return cPickle.load(handle)

    # save documents tfidf
    def save_tfidf_document(self, tfidf):
        with open(path + "/Python/bimbingan_data/tfidf-final-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def load_tfidf_document(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        #self.log("inverse_document_frequencies", all_tokens_set)
        self.save_vector_space(all_tokens_set)
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))

        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]

        return sorted_idf_values

    def tfidf(self, documents):
        tokenized_document = [self.tokenize(d[0]) for d in documents]
        idf = self.inverse_document_frequencies(tokenized_document)
        #self.log("tfidf", idf)
        tfidf_documents = []
        for document in tokenized_document:
            tfidf = []
            for term in idf.keys():
                tf = self.sublinear_term_frequency(term, document)
                tfidf.append(tf * idf[term])
            tfidf_documents.append(tfidf)

        self.save_tfidf_document(tfidf_documents)
        return tfidf_documents

similary = Similarity()
similary.DEBUG(True)

# 1. Preprocessing document and save
# we have done, so we are not run this again
similary.save_documents()

# 2. Calc TF-IDF documents and save both vector space and TF-IDF
#similary.tfidf(similary.load_documents())