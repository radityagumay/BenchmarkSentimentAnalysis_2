# http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

import nltk
import string
import sys, unicodedata
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf4():
    def __init__(self):
        self.document = []
        self.punctuation = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        self.stemmer = PorterStemmer()
        self.token_dict = {}

    def dict(self):
        for document in self.document:
            doc = document.lower()
            doc = doc.translate(self.punctuation)
            self.token_dict[document] = doc

    # remove prefix, suffix
    def stop_word(self):
        return set(stopwords.words('english'))

    # find root word
    def stem_tokens(self, tokens):
        stemmed = []
        for item in tokens:
            stemmed.append(self.stemmer.stem(item))
        return stemmed

    # split into piece of words []
    def get_tokens(self):
        # make lower all word
        index = 0
        collection = []
        for document in self.document:
            doc = document.lower()
            # remove all punctuation
            doc = doc.translate(self.punctuation)
            tokens = word_tokenize(doc)
            new_tokens = []
            for token in tokens:
                if token not in self.stop_word():
                    if token:
                        new_tokens.append(token)

            new_tokens = self.stem_tokens(new_tokens)
            collection.append((index, new_tokens))
        return collection

    def add_document(self, document):
        self.document.append(document)

    def add_documents(self, documents):
        self.document = documents

    def show_information(self):
        for collection in self.get_tokens():
            count = Counter(collection[1])
            print(count.most_common(10))

    def show_tfidf_information(self):
        self.dict()

        tfidf = TfidfVectorizer(tokenizer=self.get_tokens()[0][1])
        tfs = tfidf.fit_transform(self.token_dict)
        print(tfidf)
        print(tfs)

tfidf = tfidf4()
tfidf.add_document(
    "It was remarked in the preceding paper, that weakness and divisions at home would invite dangers from abroad; and that nothing would tend more to secure us from them than union, strength, and good government within ourselves. This subject is copious and cannot easily be exhausted.")
#tfidf.show_information()
tfidf.show_tfidf_information()
