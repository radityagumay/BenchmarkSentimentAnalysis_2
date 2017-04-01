from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import codecs, sys, glob, os, unicodedata

text = "hello world, i am good"

tokenized = word_tokenize(text)

stemmer = EnglishStemmer()

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))


# Create function with which to strip punct from unicode
# def remove_punctuation_2(term):
#     return term.translate(tbl)


def is_string_empty(string):
    if string == "":
        return False
    return True


for i in tokenized:
    stem = stemmer.stem(i)
    punc = stem.translate(tbl)
    if is_string_empty(punc):
        stop = punc not in set(stopwords.words('english'))
        print(stop, punc)



# Define english stopword
# def stop_word_2(term):
#     return term not in set(stopwords.words('english'))


# when we load data from database, we should consifer use below approach to make
# data preprocessing
# 1. load data from database
# 2. remove punctuation
# 3. define stopword


# sentence = "this is a foo bar sentence"
# print([i for i in sentence.lower().split() if i not in stop])
