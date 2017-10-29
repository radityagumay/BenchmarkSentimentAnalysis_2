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
import numpy as np
import csv

class runner:
    def load_2000_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_2000_all_tokens(self):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_tokens.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_2000_idf(self):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_idf.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_2000_documents(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def s_save_final_2000_tfidf_tokens(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_final_2000_tokens_tfidf_documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def s_load_final_2000_tfidf_tokens(self):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_final_2000_tokens_tfidf_documents.pickle",
                  "rb") as handle:
            return cPickle.load(handle)

    def s_save_final_2000_tfidf_sentence(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_final_2000_tfidf_documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def s_load_final_2000_tfidf_sentence(self):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_final_2000_tfidf_documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_2000_tfidf_documents(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_tfidf_documents.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_2000_tfidf_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_tfidf_documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_2000_all_tokens(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_tokens.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def save_2000_idf(self, data):
        with open(path + "/Python/bimbingan_data/tfidf-benchmark/all_2000_idf.pickle", "wb") as handle:
            cPickle.dump(data, handle)

    def load_scenario_3_train(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_scenario_3_test(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "rb") as handle:
            return cPickle.load(handle)

    def is_string_not_empty(self, string):
        if string == "":
            return False
        return True

    def preprocessing(self, sentences):
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
        return new_sentence

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

    def collection_tokens(self):
        train = self.load_scenario_3_train()
        test = self.load_scenario_3_test()
        collection = train + test
        tokenized_documents = []
        collection_documents = []
        for i in range(0, 2000):
            tokens = word_tokenize(self.preprocessing(collection[i][0]))
            tokenized_documents.append(tokens)
            collection_documents.append(collection[i])
        idf, all_tokens = self.inverse_document_frequencies(tokenized_documents)
        tfidf_documents = []
        index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tfidf = self.term_frequency(term, document) * idf[term]
                doc_tfidf.append(tfidf)
            print(doc_tfidf)
            tfidf_documents.append((doc_tfidf, collection[index][0], collection[index][1]))
            index += 1
        print(tfidf_documents)
        self.save(collection_documents, all_tokens, idf, tfidf_documents)

    def save(self, collection_documents, all_tokens, idf, tfidf_documents):
        self.save_2000_documents(collection_documents)
        self.save_2000_all_tokens(all_tokens)
        self.save_2000_idf(idf)
        self.write_csv(all_tokens, tfidf_documents)
        self.save_2000_tfidf_documents(tfidf_documents)

    def write_csv(self, tokens_header, data):
        out = csv.writer(open(path + "/Python/bimbingan_data/tfidf-benchmark/tfidf_2000_benchmark.csv", "w"),
                         delimiter=',', quoting=csv.QUOTE_ALL)
        out.writerow(tokens_header)
        # for d in data:
        #     out.writerow(d[0])

    def s_inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = self.all_defined_label()
        all_tokens_set = sorted(all_tokens_set)
        for tkn in all_tokens_set:
            contains_token = 0
            for t_doc in tokenized_documents:
                print(tkn, t_doc)
                if tkn in t_doc:
                    contains_token += 1
            if contains_token == 0:
                idf_values[tkn] = 0.0
            else:
                idf_values[tkn] = 1 + math.log(len(tokenized_documents) / contains_token)
        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        return sorted_idf_values, all_tokens_set

    def labeling_document(self):
        tokens = self.load_2000_all_tokens()
        documents = self.load_2000_tfidf_documents()

        for i in range(0, len(documents)):
            tfidf = documents[i][0]
            sentence = documents[i][1]
            sentiment = documents[i][2]

        preprocessing_document = []
        for d in documents:
            preprocessing_document.append(self.preprocessing(d[1]))
        tokenized_documents = []
        for t in range(0, len(preprocessing_document)):
            tokens = word_tokenize(preprocessing_document[t])
            tokenized_documents.append(tokens)
        idf, all_tokens_set = self.s_inverse_document_frequencies(tokenized_documents)
        tfidf_documents = []
        index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                if term in self.bug_label():
                    tfidf = self.term_frequency(term, document) * idf[term]
                    doc_tfidf.append((tfidf, "bug"))
                elif term in self.ui_label():
                    tfidf = self.term_frequency(term, document) * idf[term]
                    doc_tfidf.append((tfidf, "ui"))
                elif term in self.feature_label():
                    tfidf = self.term_frequency(term, document) * idf[term]
                    doc_tfidf.append((tfidf, "feature"))
            tfidf_documents.append(doc_tfidf)
            index += 1

        new_tfidf_with_label = []
        for n in tfidf_documents:
            final_label = ""
            tfidf_value = []
            for s in n:
                value = s[0]
                label = s[1]
                bug_count = 0
                ui_count = 0
                feature_count = 0
                if value > 0.0:
                    if label is "bug":
                        bug_count += 1
                    elif label is "ui":
                        ui_count += 1
                    elif label is "feature":
                        feature_count += 1

                    if bug_count > max(ui_count, feature_count):
                        final_label = "bug"
                    elif ui_count > max(bug_count, feature_count):
                        final_label = "ui"
                    elif feature_count > max(bug_count, ui_count):
                        final_label = "feature"
                    else:
                        final_label = "other"
                else:
                    c = run.random_label().split(',')
                    random.shuffle(c)
                    final_label = c[0]
                tfidf_value.append(value)
            new_tfidf_with_label.append((tfidf_value, final_label))
        final_collection = []
        for i in range(0, len(new_tfidf_with_label)):
            tfidf = new_tfidf_with_label[i][0]
            category = new_tfidf_with_label[i][1]
            sentence = documents[i][1]
            sentiment = documents[i][2]
            final_collection.append((sentence, tfidf, sentiment, category))

        self.s_save_final_2000_tfidf_tokens(all_tokens_set)
        self.s_save_final_2000_tfidf_sentence(final_collection)
        self.s_write_csv(all_tokens_set, new_tfidf_with_label)

    def s_write_csv(self, tokens_header, data):
        out = csv.writer(open(path + "/Python/bimbingan_data/tfidf-benchmark/tfidf_final_2000_benchmark.csv", "w"),
                         delimiter=',', quoting=csv.QUOTE_ALL)
        t = []
        for ts in range(0, len(tokens_header)):
            t.append(tokens_header[ts])
            if ts is 97:
                t.append('label')
        out.writerow(t)

        collection = []
        for d in range(0, len(data)):
            s = data[d][0]
            v = data[d][1]
            coll = []
            for c in range(0, len(s)):
                coll.append(s[c])
                if c is 97:
                    coll.append(v)
            collection.append(coll)
        #print(collection)
        for d in collection:
            out.writerow(d)

    def random_label(self):
        return "bug, ui, feature"

    def preprocessing_label(self, labels):
        collection  = []
        for l in labels:
            t = l.lower()
            t = stemmer.stem(t)
            if t not in collection:
                collection.append(t)
        print(collection)

    def all_defined_label(self):
        return self.bug_label() + self.ui_label() + self.feature_label()

    def ui_label(self):
        ui = ['timelin', 'icon', 'circular', 'pic', 'layout', 'display', 'black', 'white', 'horribl', 'chang', 'avatar', 'overhaul', 'overwrit', 'eyesor', 'ui', 'design', 'explor', 'tab', 'smooth', 'uniqu', 'theme', 'button']
        return ui

    def feature_label(self):
        feature = ['reset', 'password', 'upgrad', 'profil', 'chang', 'media', 'social', 'themselv', 'notif', 'photograph', 'periscop', 'perfect', 'greatdo', 'filter', 'facebook', 'entertain', 'featur', 'communic', 'brow', 'pictur', 'upload', 'video', 'camera', 'campaign', 'catagori', 'celeb', 'celebr', 'celebrati', 'changelog', 'charact', 'chat', 'cheap', 'comfort', 'configura', 'content']
        return feature

    def bug_label(self):
        bug = ['shutdown', 'disappoint', 'slow', 'worst', 'problem', 'serious', 'suck', 'applic', 'out', 'kick', 'uninst', 'annoy', 'bloodi', 'perform', 'overwhelm', 'overprocess', 'long', 'hack', 'evil', 'goddamn', 'damn', 'crash', 'forc', 'hung', 'lag', 'freez', 'speed', 'buffer', 'bug', 'buggi', 'bullshit', 'caught', 'close', 'complain', 'complaint', 'corrupt', 'crap', 'crazi', 'delay', 'destroy', 'dumb', 'error']
        return bug

run = runner()
#run.labeling_document()
print(run.s_load_final_2000_tfidf_sentence()[0])
print(run.s_load_final_2000_tfidf_tokens())

#print(run.preprocessing_label(run.bug_label()))