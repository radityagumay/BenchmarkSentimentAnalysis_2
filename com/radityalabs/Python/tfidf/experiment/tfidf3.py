from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import sys, unicodedata

stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

class tfidf:
    def __init__(self):
        self.documents = []
        self.query = ""
        self.tablePunctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

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

    def addDocumentQuery(self, query):
        self.query = query

    def calcTfIdf(self):
        print("Query : ", self.query)

    def simpleSplit(self, document):
        return document.lower().split(None)

    def calcTermFrequency(self):
        for document in self.documents:
            tokens = self.simpleSplit(document[1])
            newTokens = []
            for token in tokens:
                newTokens.append(token.translate(self.tablePunctuation))


tfidf = tfidf()
document_query = "China has shinzo abe"
document_tokens1 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_tokens2 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_tokens3 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_tokens4 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_tokens5 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_tokens6 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_tokens7 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

documents = [(1, document_tokens1), (2, document_tokens2), (3, document_tokens3), (4, document_tokens4),
             (5, document_tokens5), (6, document_tokens6), (7, document_tokens7)]

tfidf.addDocuments(documents)
tfidf.addDocumentQuery(document_query)
tfidf.calcTermFrequency()

