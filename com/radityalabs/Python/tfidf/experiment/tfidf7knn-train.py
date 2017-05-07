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

    def log(self, message):
        if self.is_debug:
            print(message)

    def load_documents(self):
        document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
        document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
        document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
        document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
        document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
        document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
        document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"
        return [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

    def load_preprocessing_document(self):
        with open(path + "/Python/bimbingan_data/tfidf-new-document-preprocessing.pickle", "rb") as handle:
            return cPickle.load(handle)

    def preprocessing_document(self):
        docs = []
        for document in self.load_documents():
            sentence = ""
            tokens = word_tokenize(document)
            for token in tokens:
                if len(token) > 3:
                    stem = self.stemmer.stem(token)
                    punct = stem.translate(self.tablePunctuation)
                    if punct is not None:
                        stop = punct not in set(stopwords.words('english'))
                        if stop:
                            sentence += str(punct.lower()) + " "
            docs.append(sentence)
        self.log(docs)
        with open(path + "/Python/bimbingan_data/tfidf-new-document-preprocessing.pickle", "wb") as handle:
                cPickle.dump(docs, handle)
                self.log("saving new document preprocessing is done")

    def load_document_query(self):
        return "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."

    def add_documents(self, documents):
        self.documents = documents

    def add_document(self, document):
        self.document = document

    def save_predefine_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-predefine.pickle", "wb") as handle:
                cPickle.dump(self.tfidf(self.load_documents()), handle)
                print("saving predefine data's is done")

    def load_predefine_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-predefine.pickle", "rb") as handle:
            return cPickle.load(handle)

    def euclidian(self, a, b):
        return distance.euclidean(a, b)

    # def euclidean_distance(self, row):
    #     inner_value = 0
    #     for k in distance_columns:
    #         inner_value += (row[k] - selected_player[k]) ** 2
    #     return math.sqrt(inner_value)

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

    def jaccard_similarity(self, query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection) / len(union)

    def term_frequency(self, term, tokenized_document):
        return tokenized_document.count(term)

    def sublinear_term_frequency(self, term, tokenized_document):
        count = tokenized_document.count(term)
        if count == 0:
            return 0
        return 1 + math.log(count)

    def augmented_term_frequency(self, term, tokenized_document):
        max_count = max([self.term_frequency(t, tokenized_document) for t in tokenized_document])
        return (0.5 + ((0.5 * self.term_frequency(term, tokenized_document)) / max_count))

    def save_vector_space(self, words):
        with open(path + "/Python/bimbingan_data/tfidf-vector-space-words.pickle", "wb") as handle:
                cPickle.dump(words, handle)
                print("saving vector space is done")

    def load_vector_space(self):
        with open(path + "/Python/bimbingan_data/tfidf-vector-space-words.pickle", "rb") as handle:
            return cPickle.load(handle)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
        return idf_values

    def inverse_document_frequencies_single(self, tokenized_document):
        idf_values = {}
        all_tokens_set = self.load_vector_space()
        all_documents = self.load_preprocessing_document()
        for tkn in all_tokens_set:
            contains_token = 0
            for document in all_documents:
                contains_token += document.count(tkn)
            idf_values[tkn] = 1 + math.log(len(tokenized_document) / (contains_token))
        return idf_values

    # dont forget to lower all word, cz calc will be case sensitive
    def tfidf(self, documents):
        self.tokenized_documents = [self.tokenize(d) for d in documents]
        idf = self.inverse_document_frequencies(self.tokenized_documents)
        tfidf_documents = []
        document_index = 0
        for document in self.tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tf = self.sublinear_term_frequency(term, document)
                doc_tfidf.append(tf * idf[term])
            tfidf_documents.append(doc_tfidf)
            document_index += 1
        return tfidf_documents

    def tfidf_query(self, document):
        tokenized_query_document = self.tokenize(document)
        idf = self.inverse_document_frequencies_single(tokenized_query_document)
        doc_query_tfidf = []
        for term in idf.keys():
            tf = self.sublinear_term_frequency(term, document)
            doc_query_tfidf.append(tf * idf[term])
        tfidf_query_document = doc_query_tfidf
        return tfidf_query_document

    def tfidf_query_with_predefine(self, query_document):
        # out predefine document with tfidf
        # [0.0 , 3.5, ...]
        local_tfidf_documents = self.load_predefine_documents()

        # ["Hello", "World"]
        tokenized_query_document = self.tokenize(query_document)
        idf = self.inverse_document_frequencies_single(tokenized_query_document)
        doc_query_tfidf = []
        for term in idf.keys():
            tf = self.sublinear_term_frequency(term, query_document)
            doc_query_tfidf.append(tf * idf[term])
        tfidf_query_document = doc_query_tfidf

    def sklearn_tfidf(self):
        sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=self.tokenize)
        return sklearn_tfidf.fit_transform(self.load_documents())

    def cosine_similarity(self, vector1, vector2):
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
        if not magnitude:
            return 0
        return dot_product / magnitude

# similarity = Similarity()
# #print(similarity.tfidf_query(similarity.load_document_query()))
#
# #similarity.tfidf(similarity.load_documents())
# #print(similarity.tfidf(similarity.load_documents()))
#
# #similarity.tfidf_query(similarity.load_document_query())
#
# #document = similarity.tfidf(similarity.load_documents())[0]
# #print(similarity.cosine_similarity(document, document))
#
# #similarity.tfidf_query_with_predefine(similarity.load_document_query())
# #print("{} {}".format(similarity.load_predefine_documents(), len(similarity.load_predefine_documents())))
#
# from collections import Counter
#
# # doc1 = "nasi goreng petaka"
# # doc2 = "nasi bebek hitam"
# #
# # number_of_occurence = 0
# # doc_coll = [doc1, doc2]
# # for sentence in doc_coll:
# #     sen = sentence.split()
# #     number_of_occurence += sen.count("kotak")
# #
# # print(number_of_occurence)
#
# tfidf_representation = similarity.tfidf(similarity.load_documents())
# query_tfidf_representation = similarity.tfidf_query(similarity.load_document_query())
# our_tfidf_comparisons = []
# for count_0, doc_0 in enumerate(tfidf_representation):
#     for count_1, doc_1 in enumerate(tfidf_representation):
#         our_tfidf_comparisons.append((similarity.cosine_similarity(doc_0, doc_1), count_0, count_1))
#
# #print(query_tfidf_representation[len(query_tfidf_representation) - 1])
# #print(query_tfidf_representation)
#
# documentt = similarity.tfidf(similarity.load_preprocessing_document())
# print(similarity.cosine_similarity(documentt[0], documentt[0]))


###########
# 1. Preprocessing all documents
# 2. Calc TF-IDF
# 3. Create vector space of tokens with []
# 4. Do a same with query

sim = Similarity()
sim.DEBUG(True)

# 1. Preprocessing all document
# we also save into pickle
sim.preprocessing_document()

