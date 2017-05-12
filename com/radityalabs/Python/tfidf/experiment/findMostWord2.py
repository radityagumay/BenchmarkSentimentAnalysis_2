from textblob import TextBlob as tb
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from nltk.stem.porter import *
from nltk.corpus import stopwords
import _pickle as cPickle
import sys, unicodedata, os
import math
import operator

path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

class CommonWord:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tablePunctuation = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    def preprocessing(self, documents):
        docs = []
        for document in documents:
            sentence = ""
            print(document)
            tokens = word_tokenize(document)
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

    def load_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)

    def term_frequency(self, term, tokenized_document):
        return tokenized_document.count(term)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        all_tokens_set = sorted(all_tokens_set)
        for key in all_tokens_set:
            contains_token = map(lambda doc: key in doc, tokenized_documents)
            idf_values[key] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        print(all_tokens_set, "\n")
        return sorted_idf_values

    def tfidf(self, documents):
        tokenized_documents = []
        for document in documents:
            tokens = word_tokenize(document[0])
            tokenized_documents.append(tokens)
        idf = self.inverse_document_frequencies(tokenized_documents)
        tfidf_documents = []
        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tf = self.term_frequency(term, document)
                doc_tfidf.append((term, tf * idf[term]))
            doc_tfidf = sorted(doc_tfidf, key=operator.itemgetter(1), reverse=True)
            tfidf = [doc_tfidf, documents[doc_index][1]]
            tfidf_documents.append(tfidf)
            doc_index += 1
        return tfidf_documents

code = CommonWord()
documents = code.load_documents()
new_documents = []
for index in range(0, 3):
    new_documents.append(documents[index])
print(code.tfidf(new_documents))