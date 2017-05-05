# http://stackoverflow.com/questions/27446204/how-do-i-use-use-the-tfidf-calculating-functions-in-scikit-learn
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from operator import itemgetter
import sys, unicodedata
import re
import collections
import math
import _pickle as cPickle
import os

# Term Frequency
# Number of occurence terms t in document i
# For example:
# we have vector = i love reading a book in evening and also drinking a milk everyday entire my life rarely
# d1 = i love reading a book
# d2 = i rarely reading a book
# d3 = i never drinking milk entire my life

# ---------------------------------------------------------------------------------------------------
# word     | i love reading a book in evening and also drinking a milk everyday entire my life rarely
# ---------------------------------------------------------------------------------------------------
# document
# d1       : 1  1   1

stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")


class termObject:
    def __init__(self, term=None, value=None):
        self.term = term
        self.value = value


class tfidf:
    def __init__(self):
        self.documents = []
        self.term = ""
        self.tablePunctuation = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        self.container = None
        self.vector_terms = []
        self.collection_documents_vector_terms = []

    def train(self):
        with open(path + "/Python/bimbingan_data/twitter_train_23536_1.pickle", "rb") as handle:
            return cPickle.load(handle)

    def test(self):
        with open(path + "/Python/bimbingan_data/twitter_test_15691_1.pickle", "rb") as handle:
            return cPickle.load(handle)

    def preprocessing(self, document):
        new_document = ""
        tokens = word_tokenize(document)
        for token in tokens:
            if len(token) > 3:
                # stem = stemmer.stem(token)
                # punct = stem.translate(tbl)

                punct = token.translate(tbl)
                if punct is not None:
                    stop = punct not in set(stopwords.words('english'))
                    if stop:
                        new_document += str(punct) + " "
        return new_document

    def loadTfIdfCollection(self):
        with open(path + "/Python/bimbingan_data/tf-idf_clean_documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def saveDocuments(self, documents):
        with open(path + "/Python/bimbingan_data/tf-idf_clean_documents.pickle", "wb") as handle:
            cPickle.dump(documents, handle)
            print("saving documents data's is done")

    def loadTfIdfDocuments(self):
        documents = self.loadTfIdfCollection()
        self.documents = documents

    def loadDocuments(self):
        collection = self.train() + self.test()
        index = 0
        for doc in collection:
            self.documents.append((index, self.preprocessing(doc[0])))
            index += 1
        self.saveDocuments(self.documents)
        print(self.documents)

    def addDocument(self, document):
        self.documents.append(document)

    def add_documents(self, documents):
        for document in documents:
            tokens = word_tokenize(document[1])
            new_doc = ""
            for token in tokens:
                if len(token) > 3:
                    stem = stemmer.stem(token)
                    punct = stem.translate(tbl)
                    if punct is not None:
                        stop = punct not in set(stopwords.words('english'))
                        if stop:
                            new_doc += str(punct.lower()) + " "
            self.documents.append(new_doc)
        self.documents = documents

    def addDocumentQuery(self, term):
        self.term = self.preprocessing(term)

    def calcTfidf(self):
        print("Query : ", self.term)

    def simpleSplit(self, document):
        return document.lower().split(None)

    def tokenCount(self, document_tokens):
        return len(document_tokens)

    def termCount(self, term, document_tokens):
        return document_tokens.count(term.lower)

    def getTokens(self, str):
        return word_tokenize(str)
        # return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())

    # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
    # IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
    def calc(self):
        # calc idf first
        word_idf = []
        for document in self.documents:
            tokens_query = self.getTokens(self.term)
            tokens_current_document = self.getTokens(document[1])

            for word in tokens_current_document:
                idf = math.log(float(1 + len(self.documents)) / (1 + tokens_query.count(word)))
                word_idf.append((word, idf))

        # calc term frequency
        container = []
        for document in self.documents:
            tokens_query = self.getTokens(self.term)
            tokens_current_document = self.getTokens(document[1])

            word_in_term = {}
            index = 0
            for word in tokens_current_document:
                tf = float(tokens_query.count(word)) / len(tokens_current_document)
                idf = word_idf[index]
                word_in_term[word] = tf * idf[1]
                index += 1

            number_of_all_term = 0.0
            for number in word_in_term.values():
                number_of_all_term += number

            container.append((document, word_in_term, number_of_all_term))
        self.container = container
        # print(self.container)

    def showInformation(self):
        sort = sorted(self.container, key=itemgetter(2), reverse=False)
        print(sort)

    def preprocessing_document_item_token(self, document):
        vector_terms = []
        tokens = word_tokenize(document[1])
        for token in tokens:
            if len(token) > 3:
                stem = stemmer.stem(token)
                punct = stem.translate(tbl)
                if punct is not None:
                    stop = punct not in set(stopwords.words('english'))
                    if stop:
                        if punct not in vector_terms:
                            vector_terms.append(punct)
        return vector_terms

    def preprocessing_token(self, documents):
        vector_terms = []
        for document in documents:
            tokens = word_tokenize(document[1])
            for token in tokens:
                if len(token) > 3:
                    stem = stemmer.stem(token)
                    punct = stem.translate(tbl)
                    if punct is not None:
                        stop = punct not in set(stopwords.words('english'))
                        if stop:
                            if punct not in vector_terms:
                                vector_terms.append(punct)
        return vector_terms

    def SortedDisplayDict(self, dict):
        return "{" + ", ".join("%r: %r" % (key, dict[key]) for key in sorted(dict)) + "}"

    # def create_vector(self):
    #     self.vector_terms = self.preprocessing_token(self.documents)
    #     print(sorted(self.vector_terms))
    #     print(len(self.vector_terms))
    #
    #     documents_vector_terms = []
    #     for document in self.documents:
    #         current_document_vector_terms = sorted(self.preprocessing_token(document))
    #         dict_document_vector_terms = {}
    #         for token_vector in self.vector_terms:
    #             for token in current_document_vector_terms:
    #                 if token == token_vector:
    #                     dict_document_vector_terms[token_vector] = 1
    #                     break
    #                 else:
    #                     dict_document_vector_terms[token_vector] = 0
    #         documents_vector_terms.append((document[0], dict_document_vector_terms))
    #     print(documents_vector_terms)

    def create_vector(self):
        self.vector_terms = self.preprocessing_token(self.documents)
        print(sorted(self.vector_terms))
        print(len(self.vector_terms))

        collection_documents_vector = []
        for document in documents:
            tokens = word_tokenize(document[1])
            vector_terms = []
            for token in tokens:
                if len(token) > 3:
                    stem = stemmer.stem(token)
                    punct = stem.translate(tbl)
                    if punct is not None:
                        stop = punct not in set(stopwords.words('english'))
                        if stop:
                            if punct not in vector_terms:
                                vector_terms.append(punct)
            collection_documents_vector.append(vector_terms)

        vector_documents = []
        for vector in collection_documents_vector:
            dict_document_vector_terms = {}
            for token_vector in self.vector_terms:
                for token in vector:
                    if token == token_vector:
                        dict_document_vector_terms[token_vector] = 1
                        break
                    else:
                        dict_document_vector_terms[token_vector] = 0
            vector_documents.append(dict_document_vector_terms)
        print(vector_documents)

    def calc_tfidf(self):
        tfidf_vector = {}
        # for item in self.collection_documents_vector_terms:
        #     print(item)


tfidf = tfidf()
document_term = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_tokens1 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_tokens2 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_tokens3 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_tokens4 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_tokens5 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_tokens6 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_tokens7 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

documents = [(0, document_tokens1), (1, document_tokens2), (2, document_tokens3), (3, document_tokens4),
             (4, document_tokens5), (5, document_tokens6), (6, document_tokens7)]

tfidf.add_documents(documents)
tfidf.create_vector()
tfidf.calc_tfidf()

# from sklearn.feature_extraction.text import TfidfVectorizer
#
# corpus = ["This is very strange", "This is very nice"]
# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print(dict(zip(vectorizer.get_feature_names(), idf)))

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import SGDClassifier
#
# corpus = ['This is the first document.',
#           'This is the second second document.',
#           'And the third one.',
#           'Is this the first document?']
# vect = TfidfVectorizer()
# X = vect.fit_transform(corpus)
# print(X.todense())
#
# y = ['Relationships', 'Games']
# model = SGDClassifier()
# print(model.fit(X, y))

