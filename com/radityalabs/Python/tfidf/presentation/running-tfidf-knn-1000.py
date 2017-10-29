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
import string
import random
from sklearn.metrics import classification_report

from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

class runner:

    def load_scenario_3_train(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_scenario_3_test(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_tfidf_217_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-217-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_tfidf_217_documents(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-217-documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_tfidf_217_vector(self):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-217-vector-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_tfidf_217_vector(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-knn/tfidf-1000-217-vector-documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_tfidf_782_documents(self):
        with open(path + "/Python/tfidf/presentation/tfidf-of-782-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_tfidf_782_documents(self, data):
        with open(path + "/Python/tfidf/presentation/tfidf-of-782-documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_tfidf_782_vector(self):
        with open(path + "/Python/tfidf/presentation/tfidf-of-782-vector-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def is_string_not_empty(self, string):
        if string == "":
            return False
        return True

    def preprocessing(self, sentences):
        documents = []
        for s in sentences:
            tfidf = s[0]
            sentence = s[1]
            sentiment = s[2]
            catgory = s[3]
            tokens = word_tokenize(sentence, language="english")
            new_sentence = ""
            for token in tokens:
                if len(token) > 3:
                    if not token.isdigit():
                        t = token.lower()
                        t = stemmer.stem(t)
                        t = t.translate(tbl)
                        if self.is_string_not_empty(t):
                            valid = t not in set(stopwords.words('english'))
                            if valid:
                                new_sentence += t + " "
            documents.append((tfidf, new_sentence, sentiment, catgory))
        return documents

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

    def run_217_tfidf(self):
        documents = self.preprocessing(self.load_tfidf_217_documents())
        tokens_collection = []
        for d in documents:
            tokens = word_tokenize(d[1])
            tokens_collection.append(tokens)
        idf, all_tokens_set = self.inverse_document_frequencies(tokens_collection)

        tfidf_documents = []
        doc_index = 0
        for document in tokens_collection:
            doc_tfidf = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)
            tfidf_documents.append((doc_tfidf, documents[doc_index][1], documents[doc_index][2], documents[doc_index][3]))
            doc_index += 1
        self.save_tfidf_217_vector(all_tokens_set)
        self.save_tfidf_217_documents(tfidf_documents)

    def S_inverse_document_frequencies(self, vector, tokens):
        idf_values = {}
        all_tokens_set = vector
        for tkn in all_tokens_set:
            contains_token = 0
            for t_doc in tokens:
                if tkn in t_doc:
                    contains_token += 1
            if contains_token == 0:
                idf_values[tkn] = 0.0
            else:
                idf_values[tkn] = 1 + math.log(len(tokens) / contains_token)
        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        print(all_tokens_set, len(all_tokens_set))
        return sorted_idf_values, all_tokens_set

    def classify(self):
        tfidf = self.load_tfidf_217_documents()
        all_tokens = self.load_tfidf_217_vector()

        load_782_document = self.load_tfidf_782_documents()
        documents = []
        for d in load_782_document:
            documents.append((d[1], d[2]))
        tfidf_documents = []
        index = 0
        for document in documents:
            tokens = word_tokenize(document[0])
            idf, all_tokens_set = self.S_inverse_document_frequencies(all_tokens, tokens)
            doc_tfidf = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)
            tfidf = [doc_tfidf, documents[index][0], documents[index][1]]
            tfidf_documents.append(tfidf)
            index += 1

    def S_preprocessing(self, sentences):
        sentence = sentences
        tokens = word_tokenize(sentence, language="english")
        new_sentence = ""
        for token in tokens:
            if len(token) > 3:
                if not token.isdigit():
                    t = token.lower()
                    t = stemmer.stem(t)
                    t = t.translate(tbl)
                    if self.is_string_not_empty(t):
                        valid = t not in set(stopwords.words('english'))
                        if valid:
                            new_sentence += t + " "
        print(new_sentence)
        return new_sentence

    def collection_tokens(self):
        train = self.load_scenario_3_train()
        test = self.load_scenario_3_test()
        collection = train + test
        tokenized_documents = []
        for i in range(0, 10):
            tokens = word_tokenize(self.S_preprocessing(collection[i][0]))
            tokenized_documents.append(tokens)
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        all_tokens_set = sorted(all_tokens_set)
        print(all_tokens_set)

    def collection_data(self):
        all_tokens = self.load_tfidf_217_vector()

        train = self.load_scenario_3_train()
        test = self.load_scenario_3_test()

        collection = train + test
        random.shuffle(collection)

        documents = []
        for i in range(0, 1000):
            documents.append((collection[i][0], collection[i][1]))
        documents = self.S_preprocessing(documents)

        tfidf_documents = []
        index = 0
        for document in documents:
            tokens = word_tokenize(document[0])
            idf, all_tokens_set = self.S_inverse_document_frequencies(all_tokens, tokens)
            doc_tfidf = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)
            tfidf = [doc_tfidf, documents[index][0], documents[index][1]]
            tfidf_documents.append(tfidf)
            index += 1
        print(tfidf_documents)

    def knn(self):
        data = []
        target = []
        labeled_tfidf = self.load_tfidf_217_documents()
        for item in labeled_tfidf:
            data.append(item[0])
            target.append(item[3])

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.4)
        my_classifier = KNeighborsClassifier(5)
        my_classifier.fit(X_train, y_train)

        for d in self.load_tfidf_782_documents():
            predictions = my_classifier.predict([d[0]])
            print(predictions)

run = runner()
run.collection_tokens()