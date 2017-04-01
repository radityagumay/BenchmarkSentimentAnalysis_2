from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import codecs, sys, glob, os, unicodedata

text = "hello world, i am good"

# when we load data from database, we should consifer use below approach to make
# data preprocessing
# 1. load data from database
# 2. tokenized
# 3. Stemming
# 4. punctuation
# 5. stopword

tokenized = word_tokenize(text)

stemmer = EnglishStemmer()

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))


def is_string_empty(string):
    if string == "":
        return False
    return True


sentence = ""

for i in tokenized:
    lower = i.lower()  # toLower().Case
    if len(lower) > 3:  # only len > 3
        stem = stemmer.stem(lower)  # root word
        punc = stem.translate(tbl)  # remove ? ! @ etc
        if is_string_empty(punc):  # check if not empty
            stop = punc not in set(stopwords.words('english'))
            if stop:  # only true we append
                sentence += str(punc) + " "

print(sentence)



# Define english stopword
# def stop_word_2(term):
#     return term not in set(stopwords.words('english'))


# sentence = "this is a foo bar sentence"
# print([i for i in sentence.lower().split() if i not in stop])
