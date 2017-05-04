# http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

import nltk
import string
import sys, unicodedata
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class tfidf4():
    def __init__(self):
        self.document = ""
        self.punctuation = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    def stop_word(self):
        return set(stopwords.words('english'))

    def get_tokens(self):
        # make lower all word
        doc = self.document.lower()
        # remove all punctuation
        doc = doc.translate(self.punctuation)
        tokens = word_tokenize(doc)
        new_tokens = []
        for token in tokens:
            if token not in self.stop_word():
                if token:
                    new_tokens.append(token)
        return new_tokens

    def add_document(self, document):
        self.document = document

    def show_information(self):
        count = Counter(self.get_tokens())
        print(count.most_common(10))

tfidf = tfidf4()
tfidf.add_document(
    "It was remarked in the preceding paper, that weakness and divisions at home would invite dangers from abroad; and that nothing would tend more to secure us from them than union, strength, and good government within ourselves. This subject is copious and cannot easily be exhausted.")
tfidf.show_information()
