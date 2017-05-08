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
import progressbar
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

    def load_train_documents(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_train_25036_3.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_test_documents(self):
        with open(path + "/Python/bimbingan_data/twitter_nltk_test_16191_3.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_documents(self):
        documents = self.preprocessing_documents(self.load_train_documents() + self.load_test_documents())
        #self.log("save_documents", documents)

        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "wb") as handle:
            cPickle.dump(documents, handle)

    def load_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)
            #self.log("load_documents",  documents)
            #return documents

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
        bar = progressbar.ProgressBar(maxval=len(all_tokens_set), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        bar_index = 0
        for tkn in all_tokens_set:
            bar_index += 1
            bar.update(bar_index)
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
        bar.finish()

        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        return sorted_idf_values

    def load_tokenized_document(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-tokenized-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_tokenized_document(self, documents):
        tok_document = []
        for document in documents:
            tokens = self.tokenize(document[0])
            print(tokens)
            tok_document.append(tokens)

        with open(path + "/Python/bimbingan_data/tfidf-final-tokenized-documents.pickle", "wb") as handle:
            cPickle.dump(tok_document, handle)

        print("\n")
        print("=================== FINISH TOKENIZED DOCUMENT ================")

    def tfidf(self):
        tokenized_document = self.load_tokenized_document() #[self.tokenize(d[0]) for d in documents
        idf = self.inverse_document_frequencies(tokenized_document)
        tfidf_documents = []
        bar = progressbar.ProgressBar(maxval=len(tokenized_document), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        bar_index = 0
        for document in tokenized_document:
            tfidf = []
            bar_index += 1
            bar.update(bar_index)
            for term in idf.keys():
                tf = self.sublinear_term_frequency(term, document)
                tfidf.append(tf * idf[term])
            tfidf_documents.append(tfidf)
        bar.finish()
        return tfidf_documents

similary = Similarity()
similary.DEBUG(True)

# 1. Preprocessing document and save
# we have done, so we are not run this again
#similary.save_documents()

# 2. Tokenzied Documents
#similary.tokenized_document(similary.load_documents())

# 3. Calc TF-IDF documents and save both vector space and TF-IDF
tfidf = similary.tfidf()
similary.save_tfidf_document(tfidf)
