from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from operator import itemgetter
import sys, unicodedata
import re
import math

stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

class termObject:
    def __init__(self, term=None, value=None):
        self.term = term
        self.value = value

class tfidf:
    def __init__(self):
        self.documents = []
        self.term = ""
        self.tablePunctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        self.container = None

    def addDocument(self, document):
        self.documents.append(document)

    def addDocuments(self, documents):
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
                            new_doc += str(punct) + " "
            self.documents.append(new_doc)
        self.documents = documents

    def addDocumentQuery(self, term):
        self.term = term

    def calcTfidf(self):
        print("Query : ", self.term)

    def simpleSplit(self, document):
        return document.lower().split(None)

    def tokenCount(self, document_tokens):
        return len(document_tokens)

    def termCount(self, term, document_tokens):
        return document_tokens.count(term.lower)

    def getTokens(self, str):
        return re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())

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

    def showInformation(self):
        sort = sorted(self.container, key=itemgetter(2), reverse=True)
        print(sort)

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

tfidf.addDocuments(documents)
tfidf.addDocumentQuery(document_term)
tfidf.calc()
tfidf.showInformation()
