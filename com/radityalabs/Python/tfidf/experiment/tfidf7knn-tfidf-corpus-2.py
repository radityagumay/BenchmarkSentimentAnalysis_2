# https://gist.githubusercontent.com/anabranch/48c5c0124ba4e162b2e3/raw/6b1acf8391b15ad3a663beb7e685e6835c964036/tfpdf.py
# http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
from __future__ import division
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import operator
import csv
import random
import _pickle as cPickle
import sys, unicodedata, os
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")


class Similarity:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tablePunctuation = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    def term_frequency(self, term, tokenized_document):
        return tokenized_document.count(term)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        all_tokens_set = sorted(all_tokens_set)
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        print(all_tokens_set)
        return sorted_idf_values

    def save_tfidf_150_documents(self, tfidf):
        with open(path + "/Python/bimbingan_data/knn/tfidf-150-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def load_tfidf_150_documents(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    # 1. preprocessing documents
    def load_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)

    # 2. manually label
    def load_document_with_categorized_label(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-manually-labeled-217-with-sentiment-category.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            collection = []
            index = 0
            for row in spamreader:
                if index != 0:
                    collection.append(', '.join(row))
                index += 1
            return collection

    # 3. calc tfidf
    """
    " [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.8134107167600364, 0.0, 0.0, 0.0, 0.0, 0.0, 5.31748811353631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.218875824868201, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.31748811353631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.064725145040942, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 6.0106352940962555, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0149030205422647, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.70805020110221, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.31748811353631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.064725145040942, 0.0, 0.0, 0.0, 0.0, 
    " 4.401197381662156, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.401197381662156, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.70805020110221, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.727975590428642, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.912023005428146, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ' neg', ' performance']]"
    """

    def tfidf(self):
        tokenized_documents = []
        documents = self.load_document_with_categorized_label()
        for document in documents:
            doc = document.split(",")
            tokens = word_tokenize(doc[0])
            tokenized_documents.append(tokens)
        idf = self.inverse_document_frequencies(tokenized_documents)
        tfidf_documents = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tf = self.term_frequency(term, document)
                doc_tfidf.append(tf * idf[term])
            doc = documents[doc_index].split(",")
            tfidf = [doc_tfidf, doc[1], doc[2]]
            tfidf_documents.append(tfidf)
            doc_index += 1
        return tfidf_documents

    def save_data_target_tfidf_for_knn(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-150-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    # 4. training and testing knn
    def knn(self):
        data = []
        target = []
        tfidf = self.tfidf()
        for item in tfidf:
            data.append(item[0])
            target.append(item[2])

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.5)
        my_classifier = KNeighborsClassifier()
        my_classifier.fit(X_train, y_train)
        print(my_classifier.predict([pick_sample[0]]))
        # predictions = my_classifier.predict(X_test)
        # print(data, accuracy_score(y_test, predictions))

    def load_unpreprocessing_train_data(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "rb") as handle:
            files = cPickle.load(handle)
            collection_positive = []
            collection_negative = []
            for file in files:
                if file[1] == 'pos':
                    collection_positive.append(file[0])
                elif file[1] == 'neg':
                    collection_negative.append(file[0])
            return collection_positive, collection_negative

    def load_unpreprocessing_test_data(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "rb") as handle:
            files = cPickle.load(handle)
            collection_positive = []
            collection_negative = []
            for file in files:
                if file[1] == 'pos':
                    collection_positive.append(file[0])
                elif file[1] == 'neg':
                    collection_negative.append(file[0])
            return collection_positive, collection_negative

    def load_collection_of_unpreprocessing_train_and_test_data(self):
        train_pos, train_neg = self.load_unpreprocessing_train_data()
        test_pos, test_neg = self.load_unpreprocessing_test_data()
        collection = train_pos + train_neg + test_pos + test_neg
        return collection


from sklearn.metrics import classification_report

class TfIdfRunner:
    def load_document_with_categorized_label_217(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-final-manually-labeled-217-with-sentiment-category.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            collection = []
            index = 0
            for row in spamreader:
                if index != 0:
                    collection.append(', '.join(row))
                index += 1
            return collection

    def term_frequency(self, term, tokenized_document):
        return tokenized_document.count(term)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        all_tokens_set = sorted(all_tokens_set)

        length = len(all_tokens_set)
        index = 0
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
            index += 1
            print("{} from {}".format(index, length))

        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]

        print(all_tokens_set)

        self.save_vector_space_terms_217_for_knn(all_tokens_set)
        return sorted_idf_values

    def tfidf(self):
        tokenized_documents = []
        documents = self.load_document_with_categorized_label_217()
        for document in documents:
            doc = document.split(",")
            tokens = word_tokenize(doc[0])
            tokenized_documents.append(tokens)
        idf = self.inverse_document_frequencies(tokenized_documents)
        tfidf_documents = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tf = self.term_frequency(term, document)
                doc_tfidf.append(tf * idf[term])
            doc = documents[doc_index].split(",")
            tfidf = [doc_tfidf, doc[1], doc[2]]
            tfidf_documents.append(tfidf)
            doc_index += 1

        # save tfidf
        self.save_tfidf_217_for_knn(tfidf_documents)
        return tfidf_documents

    def save_tfidf_217_for_knn(self, tfidf):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-217-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def load_tfidf_217_for_knn(self):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-217-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_vector_space_terms_217_for_knn(self, terms):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-217-vector-space-terms-documents.pickle", "wb") as handle:
            cPickle.dump(terms, handle)

    def load_vector_space_terms_217_for_knn(self):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-217-vector-space-terms-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_data_tfidf(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-data-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_target_tfidf(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-target-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def knn(self):

        X = [] #self.load_data_tfidf()
        y = [] #self.load_target_tfidf()

        tfidf_documents = self.load_tfidf_217_for_knn()
        for index in range(0, len(tfidf_documents)):
            X.append(tfidf_documents[index][0])
            y.append(tfidf_documents[index][2])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)
    
        predictions = my_classifier.predict(X_test)
        print(accuracy_score(y_test, predictions))
        new_y_test = []
        for index in range(0, len(y_train)):
            new_y_test.append(y[index])
        print(classification_report(y_train, y_train, target_names=['ui', 'bug', 'feature']))

    # 1000
    def tfidf_1000_preprocessing_documents(self):
        # tfidf_217_documents = self.load_tfidf_217_for_knn()
        # print(tfidf_217_documents)

        tfidf_217_vector_space = self.load_vector_space_terms_217_for_knn()
        print(tfidf_217_vector_space)

    def load_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)

runner = TfIdfRunner()
#runner.knn()
runner.tfidf_1000_preprocessing_documents()

# iris = datasets.load_iris()
#
# X = iris.data
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
#
# my_classifier = KNeighborsClassifier(5)
# my_classifier.fit(X_train, y_train)
#
# predictions = my_classifier.predict(X_test)
#
# print(my_classifier.predict([[6.7, 3.1, 5.6, 2.4]]))
# print(accuracy_score(y_test, predictions))
# target_names = ["1", "2", "3"]
# print(classification_report(y_train, y_test, target_names=target_names))