import _pickle as cPickle
import math
import sys, unicodedata, os
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import csv
import random
from sklearn.metrics import classification_report

path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)
nltk_stopwords = nltk.corpus.stopwords.words('english')

class TfIdf:
    def __init__(self):
        self.data = Data()

    def term_frequency(self, term, tokenized_document):
        return tokenized_document.count(term)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        all_tokens_set = sorted(all_tokens_set)

        index = 0
        length_documents = len(all_tokens_set)
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))

            index += 1
            print("{} from {}".format(index, length_documents))

        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        print(all_tokens_set)

        return sorted_idf_values, all_tokens_set

    def tfidf(self, documents):
        tokenized_documents = []

        index = 0
        length_document = len(documents)
        for document in documents:
            tokens = word_tokenize(document[0])
            tokenized_documents.append(tokens)

            index += 1
            print("{} from {}".format(index, length_document))

        idf, all_tokens_set = self.inverse_document_frequencies(tokenized_documents)

        self.data.save_vector_tokens(all_tokens_set)

        tfidf_documents = []
        tfidf_documents_with_term = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            doc_tfidf_with_term = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)

                doc_tfidf_with_term.append((term, tfidf))
            # Plain
            tfidf = [doc_tfidf, documents[doc_index][0], documents[doc_index][1]]
            tfidf_documents.append(tfidf)

            # with term
            tfidf_with_term = [doc_tfidf_with_term, documents[doc_index][0], documents[doc_index][1]]
            tfidf_documents_with_term.append(tfidf_with_term)
            doc_index += 1
            print("{} from {}".format(doc_index, length_document))

        self.data.save_tfidf_documents_with_term(tfidf_documents_with_term)
        self.data.save_tfidf_documents(tfidf_documents)
        return tfidf_documents, tfidf_documents_with_term

    def tfidf_bigram_217(self, documents):
        index = 0
        tokenized_documents = []
        length_document = len(documents)
        for document in documents:
            tokens = tokenizer.tokenize(document[0])
            bi_tokens = bigrams(tokens)
            bi_tokens = [' '.join(token).lower() for token in bi_tokens]
            bi_tokens = [token for token in bi_tokens if token not in nltk_stopwords]
            tokenized_documents.append(bi_tokens)

            index += 1
            print("{} from {}".format(index, length_document))

        idf, all_tokens_set = self.inverse_document_frequencies(tokenized_documents)
        self.data.save_vector_tokens_217(all_tokens_set)

        tfidf_documents = []
        tfidf_documents_with_term = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            doc_tfidf_with_term = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)

                doc_tfidf_with_term.append((term, tfidf))
            # Plain
            tfidf = [doc_tfidf, documents[doc_index][0], documents[doc_index][1], documents[doc_index][2]]
            tfidf_documents.append(tfidf)

            # with term
            tfidf_with_term = [doc_tfidf_with_term, documents[doc_index][0], documents[doc_index][1], documents[doc_index][2]]
            tfidf_documents_with_term.append(tfidf_with_term)
            doc_index += 1
            print("{} from {}".format(doc_index, length_document))

        self.data.save_tfidf_documents_with_term_217(tfidf_documents_with_term)
        self.data.save_tfidf_documents_217(tfidf_documents)
        return tfidf_documents, tfidf_documents_with_term

    def tfidf_bigram(self, documents):
        index = 0
        tokenized_documents = []
        length_document = len(documents)
        for document in documents:
            tokens = tokenizer.tokenize(document[0])
            bi_tokens = bigrams(tokens)
            bi_tokens = [' '.join(token).lower() for token in bi_tokens]
            bi_tokens = [token for token in bi_tokens if token not in nltk_stopwords]
            tokenized_documents.append(bi_tokens)

            index += 1
            print("{} from {}".format(index, length_document))

        idf, all_tokens_set = self.inverse_document_frequencies(tokenized_documents)

        self.data.save_vector_bigram_tokens(all_tokens_set)

        tfidf_documents = []
        tfidf_documents_with_term = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            doc_tfidf_with_term = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)

                doc_tfidf_with_term.append((term, tfidf))
            # Plain
            tfidf = [doc_tfidf, documents[doc_index][0], documents[doc_index][1]]
            tfidf_documents.append(tfidf)

            # with term
            tfidf_with_term = [doc_tfidf_with_term, documents[doc_index][0], documents[doc_index][1]]
            tfidf_documents_with_term.append(tfidf_with_term)
            doc_index += 1
            print("{} from {}".format(doc_index, length_document))

        self.data.save_tfidf_bigram_documents_with_term(tfidf_documents_with_term)
        self.data.save_tfidf_bigram_documents(tfidf_documents)
        return tfidf_documents, tfidf_documents_with_term

class Data:
    # load 41227 preprocessing document [neg, pos]
    def load_preprocessing_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)

    # ================= TFIDF =====================
    def save_tfidf_documents(self, documents):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents.pickle", "wb") as handle:
            cPickle.dump(documents, handle)

    def load_tfidf_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents.pickle", "rb") as handle:
            return cPickle.load(handle)
    # ================= TFIDF =====================

    # ================= VECTOR BIGRAM SPACE TOKENS =====================
    def save_vector_bigram_tokens(self, tokens):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-all-vector-bigram-space-tokens-documents.pickle", "wb") as handle:
            cPickle.dump(tokens, handle)

    def load_vector_bigram_tokens(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-all-vector-bigram-space-tokens-documents.pickle",
                  "rb") as handle:
            return cPickle.load(handle)
    # ================= VECTOR BIGRAM SPACE TOKENS =====================

    # ================= VECTOR SPACE TOKENS =====================
    def save_vector_tokens(self, tokens):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-all-vector-space-tokens-documents.pickle", "wb") as handle:
            cPickle.dump(tokens, handle)

    def load_vector_tokens(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-all-vector-space-tokens-documents.pickle", "rb") as handle:
            return cPickle.load(handle)
    # ================= VECTOR SPACE TOKENS =====================

    # ================= TFIDF WITH TERM =====================
    def save_tfidf_documents_with_term(self, documents):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-with-terms.pickle", "wb") as handle:
            cPickle.dump(documents, handle)

    def load_tfidf_documents_with_term(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-with-terms.pickle", "rb") as handle:
            return cPickle.load(handle)
    # ================= TFIDF WITH TERM =====================

    # ================= TFIDF BIGRAM WITH TERM =====================
    def save_tfidf_bigram_documents_with_term(self, documents):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-bigram-with-terms.pickle", "wb") as handle:
            cPickle.dump(documents, handle)

    def load_tfidf_bigram_documents_with_term(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-bigram-with-terms.pickle", "rb") as handle:
            return cPickle.load(handle)
    # ================= TFIDF BIGRAM WITH TERM =====================

    # ================= TFIDF BIGRAM =====================
    def save_tfidf_bigram_documents(self, documents):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-bigram.pickle", "wb") as handle:
            cPickle.dump(documents, handle)

    def load_tfidf_bigram_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-bigram.pickle", "rb") as handle:
            return cPickle.load(handle)
    # ================= TFIDF BIGRAM =====================

    def load_tfidf_with_labeled_category(self):
        with open(path + "/Python/bimbingan_data/tfidf/tfidf-217-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_corpus_with_sentiment_and_category(self):
        with open(path + "/Python/bimbingan_data/knn/tfidf-final-manually-labeled-217-with-sentiment-category.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            collection = []
            index = 0
            for row in spamreader:
                if index != 0:
                    collection.append(', '.join(row))
                index += 1
            return collection

    def save_vector_tokens_217(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-all-vector-space-tokens-217-documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def save_tfidf_documents_with_term_217(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-documents-with-217-terms.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def save_tfidf_documents_217(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-217-documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_tfidf_documents_217(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-217-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_training_testing_labeled_category(self, documents):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-manually-labeled.csv", 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['document', 'tfidf', 'label_sentiment', 'label_category'])
            for item in documents:
                writer.writerow([item[0], item[1], item[2], item[3]])

class Knn:
    def __init__(self):
        self.tfidf = TfIdf()
        self.data = Data()

    def run_tfidf_bigram_217(self):
        documents = self.data.load_corpus_with_sentiment_and_category()
        new_documents = []
        for index in documents:
            line = index.split(',')
            document = line[0]
            sentiment_label = line[1]
            category_label = line[2]
            new_documents.append((document, sentiment_label, category_label))

        tfidf_bigram_documents, tfidf_bigram_documents_with_term = self.tfidf.tfidf_bigram_217(new_documents)
        print(tfidf_bigram_documents)

    def run_tfidf_bigram(self):
        documents = self.data.load_preprocessing_documents()

        # random collection
        random.shuffle(documents)

        doc_range = []
        for index in range(0, (1000 - 1)):
            doc_range.append(documents[index])

        tfidf_bigram_documents, tfidf_bigram_documents_with_term = self.tfidf.tfidf_bigram(doc_range)
        print(tfidf_bigram_documents)

    def run_tfidf(self):
        # load preprocessing data
        documents = self.data.load_preprocessing_documents()

        # random collection
        random.shuffle(documents)

        #pick few datas
        doc_range = []
        for index in range(0, (1000 - 1)):
           doc_range.append(documents[index])

        tfidf_documents, tfidf_documents_with_term = self.tfidf.tfidf(doc_range)
        print(tfidf_documents)

    def classify_review(self):
        data = []
        target = []

        labeled_tfidf = self.data.load_tfidf_with_labeled_category()
        for item in labeled_tfidf:
            data.append(item[0])
            target.append(item[2])

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.4)
        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)

        tfidf = self.data.load_tfidf_documents()

        collection = []

        print(target)

        #for item in tfidf:
            #result = my_classifier.predict([item[0]])
            #collection.append((item[1], item[0], item[2], result))

        #self.data.save_training_testing_labeled_category(collection)

    def classify(self):
        data = []
        target = []

        labeled_tfidf = self.data.load_tfidf_with_labeled_category()
        for item in labeled_tfidf:
            data.append(item[0])
            target.append(item[2])

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.4)
        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)
        predictions = my_classifier.predict(X_test)

        print(" ====== KNN Classification ======== ")
        print(accuracy_score(y_test, predictions))

        target_names = ["bug", "ui", "feature"]
        print(classification_report(y_test, predictions, target_names=target_names))

    def classify_bigram(self):
        data = []
        target = []

        labeled_tfidf = self.data.load_tfidf_documents_217()
        for item in labeled_tfidf:
            data.append(item[0])
            target.append(item[3])

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.4)
        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)
        predictions = my_classifier.predict(X_test)

        print(" ====== KNN Classification with Bigram ======== ")
        print(accuracy_score(y_test, predictions))

        target_names = ["bug", "ui", "feature"]
        print(classification_report(y_test, predictions, target_names=target_names))

class Log:
    def __init__(self):
        self.data = Data()

    def print_tfidf_documents_with_term(self):
        documents = self.data.load_tfidf_documents_with_term()
        print(documents[0], len(documents))

    def print_tfidf_documents(self):
        documents = self.data.load_tfidf_documents()
        print(documents[0], len(documents))

    def print_tfidf_vector_space_terms(self):
        documents = self.data.load_vector_tokens()
        print(documents, len(documents))

    def print_tfidf_vector_bigram_space_terms(self):
        documents = self.data.load_vector_bigram_tokens()
        print(documents, len(documents))

    def print_tfidf_bigram_documents(self):
        print(self.data.load_tfidf_bigram_documents())

    def print_tfidf_bigram_documents_with_term(self):
        documents = self.data.load_tfidf_bigram_documents_with_term()
        print(documents[0], len(documents))

    def print_tfidf_with_labeled_category(self):
        print(self.data.load_tfidf_with_labeled_category())

    def print_corpus_with_sentiment_and_category(self):
        print(self.data.load_corpus_with_sentiment_and_category())

#log = Log()
#log.print_tfidf_bigram_documents()

# log = Data()
# print(log.load_corpus_with_sentiment_and_category())










