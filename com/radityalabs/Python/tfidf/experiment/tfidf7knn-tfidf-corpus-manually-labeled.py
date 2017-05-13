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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
#
# my_classifier = KNeighborsClassifier()
# my_classifier.fit(X_train, y_train)
#
# predictions = my_classifier.predict(X_test)
#
# print(my_classifier.predict([[6.7, 3.1, 5.6, 2.4]]))
# print(accuracy_score(y_test, predictions))

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
        self.save_vector_space_terms_41227(all_tokens_set)
        self.save_idf_41227(sorted_idf_values)

        return sorted_idf_values

    def save_vector_space_terms_41227(self, tfidf):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-vector-space-41227.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def save_idf_41227(self, idf):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-idf-41227.pickle", "wb") as handle:
            cPickle.dump(idf, handle)

    def save_tfidf_150_documents(self, tfidf):
        with open(path + "/Python/bimbingan_data/knn/tfidf-150-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def load_tfidf_150_documents(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    # 1. preprocessing documents
    def load_documents_with_preprocessing_and_sentiment_label(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)

    # 2. manually label
    def load_document_with_categorized_label(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-2-with-sentiment-category.csv", 'r') as csvfile:
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

    def save_data_tfidf_for_knn(self, data):
        with open(path + "/Python/bimbingan_data/knn/tfidf-data-150-documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_data_tfidf_for_knn(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-data-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_target_tfidf_for_knn(self, tfidf):
        with open(path + "/Python/bimbingan_data/knn/tfidf-target-150-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def load_target_tfidf_for_knn(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-target-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_sentiment_label_tfidf_for_knn(self, tfidf):
        with open(path + "/Python/bimbingan_data/knn/tfidf-sentiment-label-150-documents.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def load_sentiment_label_tfidf_for_knn(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-sentiment-label-150-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    # 4. training and testing knn
    def knn(self):
        data = []
        sentiment = []
        target = []
        tfidf = self.tfidf()
        for item in tfidf:
            data.append(item[0])
            sentiment.append(item[1])
            target.append(item[2])

        self.save_data_tfidf_for_knn(data)
        self.save_target_tfidf_for_knn(target)
        self.save_sentiment_label_tfidf_for_knn(sentiment)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.5)
        #self.chart(X_train, X_test, y_train, y_test)
        my_classifier = KNeighborsClassifier()
        my_classifier.fit(X_train, y_train)
        predictions = my_classifier.predict(X_test)
        print(data, accuracy_score(y_test, predictions))

    def knn_with_labeled_category(self):
        data = self.load_data_tfidf_for_knn()
        sentiment = self.load_sentiment_label_tfidf_for_knn()
        target = self.load_target_tfidf_for_knn()
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.5)
        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)
        predictions = my_classifier.predict(X_test)
        print(data, accuracy_score(y_test, predictions))

    """ 
    " return will tuple 
    " 1: tfidf, 2: document, 3: sentiment label
    " 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'nice like thi full ', 'pos']
    """
    def knn_classification(self):
        # 1. Load Documents
        documents = self.load_documents_with_preprocessing_and_sentiment_label()

        # 2. Select Only 1000 Documents
        random.shuffle(documents)
        new_documents = []
        for index in range(0, 100):
            new_documents.append(documents[index])

        # 3. Tf-Idf
        tokenized_documents = []
        for document in new_documents:
            tokens = word_tokenize(document[0])
            tokenized_documents.append(tokens)

        idf = self.inverse_document_frequencies(tokenized_documents)
        tfidf_documents = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tf = self.term_frequency(term, document)
                doc_tfidf.append(tf * idf[term])
            doc = new_documents[doc_index]
            tfidf = [doc_tfidf, doc[0], doc[1]]
            tfidf_documents.append(tfidf)
            doc_index += 1

        # 4. knn
        data = self.load_data_tfidf_for_knn()
        target = self.load_target_tfidf_for_knn()
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.5)
        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)

        # iteration
        for item in tfidf_documents:
            print(len(data[0]), len(item[0]))
            predictions = my_classifier.predict([item[0]])
            print(predictions, item[1], item[2])

        # save result

    def save_knn_1000_documents(self, knn):
        with open(path + "/Python/bimbingan_data/knn/tfidf-knn-1000-documents.pickle", "wb") as handle:
            cPickle.dump(knn, handle)

    def chart(self, X_train, X_test, y_train, y_test):
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        h = .02
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            classifier = KNeighborsClassifier(5, weights=weights)
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            print(classifier, accuracy_score(y_test, predictions))

            # clf = neighbors.KNeighborsClassifier(5, weights=weights)
            # clf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (5, weights))

        plt.show()

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

    def load_preprocessing_document_with_tfidf(self):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tfidf-sample.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_preprocessing_document_tfidf(self, tfidf):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-preprocessing-documents-41227.pickle", "wb") as handle:
            cPickle.dump(tfidf, handle)

    def training_and_testing_tfidf_knn_manual_labeled_category(self):
        # 1. Load Documents
        documents = self.load_documents_with_preprocessing_and_sentiment_label()
        """
        " forc close onc updat header automat close cant even look wall caus twitter isnt respond tweet anyway pleas thank , 
        " neg, 
        """
        tokenized_documents = []
        for document in documents:
            tokens = word_tokenize(document[0])
            tokenized_documents.append(tokens)

        idf = self.inverse_document_frequencies(tokenized_documents)
        # tfidf_documents = []
        # doc_index = 0
        # for document in tokenized_documents:
        #     doc_tfidf = []
        #     for term in idf.keys():
        #         tf = self.term_frequency(term, document)
        #         doc_tfidf.append(tf * idf[term])
        #     doc = new_documents[doc_index]
        #     tfidf = [doc_tfidf, doc[0], doc[1]]
        #     tfidf_documents.append(tfidf)
        #     doc_index += 1



        # # 2. Select Only 1000 Documents
        # random.shuffle(documents)
        # new_documents = []
        # for index in range(0, 100):
        #     new_documents.append(documents[index])
        #
        # # 3. Tf-Idf
        # tokenized_documents = []
        # for document in new_documents:
        #     tokens = word_tokenize(document[0])
        #     tokenized_documents.append(tokens)
        #
        # idf = self.inverse_document_frequencies(tokenized_documents)
        # tfidf_documents = []
        # doc_index = 0
        # for document in tokenized_documents:
        #     doc_tfidf = []
        #     for term in idf.keys():
        #         tf = self.term_frequency(term, document)
        #         doc_tfidf.append(tf * idf[term])
        #     doc = new_documents[doc_index]
        #     tfidf = [doc_tfidf, doc[0], doc[1]]
        #     tfidf_documents.append(tfidf)
        #     doc_index += 1
        #
        # # 4. knn
        # data = self.load_data_tfidf_for_knn()
        # target = self.load_target_tfidf_for_knn()
        # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.5)
        # my_classifier = KNeighborsClassifier(5)
        # my_classifier.fit(X_train, y_train)
        #
        # # iteration
        # for item in tfidf_documents:
        #     print(len(data[0]), len(item[0]))
        #     predictions = my_classifier.predict([item[0]])
        #     print(predictions, item[1], item[2])

        # save result

    # Load Labeled Categorised [1] Bug, [2] UI, [3] Feature, [4] performance
    def load_corpus_with_sentiment_and_category(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-final-corpus-2-with-sentiment-category.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            collection = []
            index = 0
            for row in spamreader:
                if index != 0:
                    collection.append(', '.join(row))
                index += 1
            return collection

    # Manually Labeled Categorised [1] Bug, [2] UI, [3] Feature, [4] performance
    def save_corpus_with_sentiment_and_category(self):
        range_documents = []
        documents = self.load_documents_with_preprocessing_and_sentiment_label()
        for index in range(149, 1000):
            range_documents.append((documents[index], "undefine"))
        with open(path + "/Python/bimbingan_data/knn/tfidf-manually-labeled.csv", 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['document', 'label_sentiment', 'label_category'])
            for item in range_documents:
                document = item[0][0]
                label_sentiment = item[0][1]
                label_category = item[1]
                writer.writerow([document, label_sentiment, label_category])

code = Similarity()
code.training_and_testing_tfidf_knn_manual_labeled_category()
