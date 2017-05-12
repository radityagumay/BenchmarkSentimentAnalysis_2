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
import _pickle as cPickle
import sys, unicodedata, os
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

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

    # preprocessing documents
    def load_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-preprocessing-document.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_twitter_query(self):
        return "Stop the updates Twitter is becoming more like facebook everyday. We already have a Facebook & thats not even right. If I wanna use Facebook Ill just log on to it & uninstall twitter. Messing up again"

    def load_twitter_documents(self):
        document_0 = "Force close Once I updated my header it just automatically closing down and I cant even look at my wall cause Twitter isnt responding but I can tweet anyway please fix. Thanks"
        document_1 = "Draft button media viewing style! 1)PUT THE DRAFT BUTTON BACK AS AN OPTION ON THE 3DOTS AT THE TL PAGE!!! 2)FIX THE STYLE THE MEDIA IS SEEN! MULTIPLE IN A ROW NOT SCROLL DOWN FOR MORE!??????? *DESIGN WISE EACH UPDATE GETS WORSE!!!!"
        document_2 = "Ok One less star because muted users still appear on my lists when they are re-tweeted and the fact that I cant seem to be able to mute hashtags. Another star removed because you either dont care about your users or are too lazy to update the Whats new section."
        document_3 = "Priorities If you follow a lot of pages and friends its very hard to notice all of your friends tweets. I think Twitter should prioritize tweets from friends then only pages. Only through this you can catch up on what you missed. Or a priority settings... Notifications for tweets from starred accounts sometimes dont even pop up at all. Please fix this issue as soon as possible."
        document_4 = "Hearts... Not really When I hit the heart button the app crashes immediately its been like that ever since the introduction of it. Maybe its time to finally do something? After MONTHS?"
        document_5 = "Poor system on phone verification My first account has been locked and so i followed your instruction but i could not proceed to the next step since ive been receiving the code from u super late and when i entered the code the page already expired(found error 404). I then decided to create a new account since yesterday but still unable to sign it up until now. Been posting my cp no. twice for the phone verification section but couldnt receive any code until now.As per your articlewell wait the code for just few minutes.pls fix it."
        document_6 = "I cannot see any videos The app closes or it brings me back to my TL if Im viewing someones profile. "
        document_7 = "You guys suck thats all. Too much useless and frequent updates. Please keep the app stable first before releasing it to the public. Plus please update the version notes on the whats new? section. Hate to see the weve add comments bla bla bla. Please be serious about this guys."
        document_8 = "No change after many updates Followers are still not in any kind of order... for months now... new followers are in the middle and old followers are at the top... not just me but anyone I follow. Also videos play very sluggish if at all "
        document_9 = "tweets take forever to send Have to say not surprising the amount of updates that seem to to nothingOne day Twitter will do an updateand actually make it betterSadly probably wont be in our lifetimes. jumps so u have to scroll bk down to try find tweet FFS how long have people complaind about this and still it does it."
        return [[document_0, "bug"], [document_1, "ui"], [document_2, "feature"], [document_3, "ui"],
                [document_4, "bug"], [document_5, "bug"], [document_6, "bug"], [document_7, "feature"],
                [document_8, "feature"], [document_9, "feature"]]

    def term_frequency(self, term, tokenized_document):
        return tokenized_document.count(term)

    def inverse_document_frequencies(self, tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        all_tokens_set = sorted(all_tokens_set)
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]

        print(all_tokens_set)
        #self.save_twitter_tokens(all_tokens_set)
        return sorted_idf_values

    def query_inverse_document_frequencies(self, query_tokenized):
        idf_values = {}

        all_tokens_set = self.load_twitter_tokens()

        for token in all_tokens_set:
            contains_token = 0
            loaded_documents = self.load_twitter_documents()
            for document in loaded_documents:
                contains_token += document[0].count(token)
                if contains_token == 0:
                    idf_values[token] = 0.0
                else:
                    idf_values[token] = 1 + math.log(len(loaded_documents) / (contains_token))

        for token in all_tokens_set:
            if not idf_values.get(token):
                idf_values[token] = 0.0

        sorted_idf_values = {}
        for key in sorted(idf_values):
            sorted_idf_values[key] = idf_values[key]
        return sorted_idf_values

    def cosine_similarity(self, vector1, vector2):
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
        if not magnitude:
            return 0
        return dot_product / magnitude

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

    def documents_twitter_tfidf(self, documents):
        tokenized_documents = []
        for document in documents:
            tokens = word_tokenize(document[0])
            tokenized_documents.append(tokens)

        self.save_twitter_tokenized_documents(tokenized_documents)

        idf = self.(tokenized_documents)
        tfidf_documents = []

        doc_index = 0
        for document in tokenized_documents:
            doc_tfidf = []
            for term in idf.keys():
                tf = self.term_frequency(term, document)
                doc_tfidf.append((term, tf * idf[term]))
            tfidf = [doc_tfidf, documents[doc_index][1]]
            tfidf_documents.append(tfidf)
            doc_index += 1
        self.save_twitter_tfidf_with_term(tfidf_documents)
        return tfidf_documents

    def query_documents_twitter_tfidf(self, query_document):
        query_token = word_tokenize(query_document)
        idf = self.query_inverse_document_frequencies(query_token)

        doc_tfidf = []
        #doc_tfidf_term = []
        for term in idf.keys():
            tf = self.term_frequency(term, query_token)
            doc_tfidf.append(tf * idf[term])
            #doc_tfidf_term.append((term, tf * idf[term]))

        #print(doc_tfidf_term)
        return doc_tfidf

    def save_twitter_tokenized_documents(self, tokens):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tokenized-documents.pickle", "wb") as handle:
            cPickle.dump(tokens, handle)

    def load_twitter_tokenized_documents(self):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tokenized-documents.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_twitter_tfidf(self, tokens):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tfidf-sample.pickle", "wb") as handle:
            cPickle.dump(tokens, handle)

    def load_twitter_tfidf(self):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tfidf-sample.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_twitter_tfidf_with_term(self, tokens):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tfidf-with-terms.pickle", "wb") as handle:
            cPickle.dump(tokens, handle)

    def load_twitter_tfidf_with_term(self):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tfidf-with-terms.pickle", "rb") as handle:
            return cPickle.load(handle)

    def save_twitter_tokens(self, tokens):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tokens.pickle", "wb") as handle:
            cPickle.dump(tokens, handle)

    def load_twitter_tokens(self):
        with open(path + "/Python/bimbingan_data/tfidf-twitter-tokens.pickle", "rb") as handle:
            return cPickle.load(handle)

    def load_corpus_with_sentiment_and_category(self):
        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-with-sentiment.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            collection = []
            index = 0
            for row in spamreader:
                if index != 0:
                    collection.append(', '.join(row))
                index += 1
            return collection

    def save_corpus_with_sentiment_and_category(self):
        range_documents = []
        preprocessing_documents = code.load_documents()

        for index in range(0, 20):
            range_documents.append((preprocessing_documents[index], "Economy"))

        with open(path + "/Python/bimbingan_data/tfidf-final-corpus-with-sentiment.csv", 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['document', 'label_sentiment', 'label_category'])
            for item in range_documents:
                document = item[0][0]
                label_sentiment = item[0][1]
                label_category = item[1]
                writer.writerow([document, label_sentiment, label_category])

code = Similarity()

# 1. we have preprocessing documents with sentiment label and category label
def tfidf():
    tokenized_documents = []
    documents = code.load_corpus_with_sentiment_and_category()
    for document in documents:
        doc = document.split(",")
        tokens = word_tokenize(doc[0])
        tokenized_documents.append(tokens)

    idf = code.inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    doc_index = 0
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = code.term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])

        doc = documents[doc_index].split(",")
        tfidf = [doc_tfidf, doc[1], doc[2]]
        tfidf_documents.append(tfidf)
        doc_index += 1
    return tfidf_documents

tfidf_representation = tfidf()

data = []
target = []

for tfidf in tfidf_representation:
    data.append(tfidf[0])
    target.append(tfidf[2])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size= .5)

my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(data, accuracy_score(y_test, predictions))

#documents = code.load_corpus_with_sentiment_and_category()

# print(documents[18].split(",")[0], my_classifier.predict([data[18]]))
# print(accuracy_score(y_test, predictions))

# tfidf = code.documents_twitter_tfidf(code.preprocessing_documents(code.load_twitter_documents()))
# print(tfidf)
# print(code.cosine_similarity(tfidf[0][0], tfidf[2][0]))

# doc = code.load_twitter_documents()[1][0]
#query_doc = code.load_twitter_query()

# print(code.load_twitter_tokens())
# print(query_doc)

# tfidf_query = code.query_documents_twitter_tfidf(query_doc)
# tfidf_documents = code.load_twitter_tfidf()

# print(tfidf_query)
# print(tfidf_documents)
#
# for document in tfidf_documents:
#     print(code.cosine_similarity(document[0], tfidf_query), document[1])

# print(code.load_twitter_tfidf_with_term())
# print(code.load_twitter_tokens())

# data = []
# target = []
#
# for document in tfidf_documents:
#     data.append(document[0])
#     target.append(document[1])
#
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size= .5)
#
# my_classifier = KNeighborsClassifier()
# my_classifier.fit(X_train, y_train)
#
# predictions = my_classifier.predict(X_test)
#
# print(tfidf_query, len(tfidf_query))
# print(data[0], len(data[0]))
#
#
# print(my_classifier.predict([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6094379124341005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.302585092994046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
# print(accuracy_score(y_test, predictions))