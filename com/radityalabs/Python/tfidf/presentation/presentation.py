import _pickle as cPickle
import math
import sys, unicodedata, os
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import csv
import random
from sklearn.metrics import classification_report

path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)
nltk_stopwords = nltk.corpus.stopwords.words('english')

def collection():
    datas = []
    datas.append("'It''s a very nice app to use, when it''s working correctly. Been having trouble with it as of late. Trying to get my notifications turned on. It keeps telling me twitter has stopped (Report) (OK). I''ve had to uninstall it and then reinstall it a number of times. If it starts working correctly I''d be very happy to give it a higher rating. Like i mentioned it is a very nice app, when working correctly. After just UNISTALLING the APP & then REINSTALLING it. It seems to be working better. I''ve had to do that about 4 or 5 times so far. ?? '")
    datas.append("'So here is what I hate: when I go to someone''s profile neither on tweets nor on tweets and replies I can't see all of their tweets!! I feel like it''s selected or something but I WANNA SEE ALL TWEETS PLEASE!! Lately I have also noticed that I do not see all the tweets of people I follow on my time line but I WANNA SEE ALL!!! also I wish I could download not only pictures in tweets but also gifs. Oh and sometimes I don''t get any notifications and sometimes I do I don''t know why but it''s annoying, please fix! I love the concept and the fact that it''s the only social media staying itself and not going Facebook or Snapchat with the stories and stuff '")
    datas.append("'Every time they so-called update the app, they add more problems than they solve. Matter how of fact, you don''t even know how they made better. So now, the search button which usually shows trending topics isn''t doing that anymore. And a tweet with a 1000 rt only shows 1 when seen among other tweets. '")
    datas.append("'Trending topics revision sucks with latest update... not as accessible and less robust. Likes and retweets counters are cut off if over 99 so 100 shows as 1 unless you isolate the tweet. Other than that, I''m mostly fine with newer changes. Showing who is replying to whom is helpful but obnoxious. Can''t think of a better option, but it''s unpleasant as is. '")
    return datas

def S_term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def S_inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    all_tokens_set = sorted(all_tokens_set)

    index = 0
    length_documents = len(all_tokens_set)
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))

        index += 1
        print("{} from {}".format(index, length_documents))

    sorted_idf_values = {}
    for key in sorted(idf_values):
        sorted_idf_values[key] = idf_values[key]
    print(all_tokens_set, len(all_tokens_set))

    return sorted_idf_values, all_tokens_set

from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
stemmer = EnglishStemmer()
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

def is_string_not_empty(string):
    if string == "":
        return False
    return True

def preprocessing(sentences):
    documents = []
    for sentence in sentences:
        tokens = word_tokenize(sentence, language="english")
        new_sentence = ""
        for token in tokens:
            if len(token) > 3:
                if not token.isdigit():
                    t = token.lower()
                    t = stemmer.stem(t)
                    t = t.translate(tbl)
                    if is_string_not_empty(t):
                        valid = t not in set(stopwords.words('english'))
                        if valid:
                            new_sentence += t + " "
        documents.append(new_sentence)
    return documents

def runner():
    documents = preprocessing(collection())

    tokenized_documents = []
    index = 0
    length_document = len(documents)
    for document in documents:
        tokens = word_tokenize(document)
        tokenized_documents.append(tokens)
        index += 1
        print("{} from {}".format(index, length_document))

    idf, all_tokens_set = S_inverse_document_frequencies(tokenized_documents)

    tfidf_documents = []
    tfidf_documents_with_term = []
    doc_index = 0
    for document in tokenized_documents:
        doc_tfidf = []
        doc_tfidf_with_term = []
        for term in idf.keys():
            tfidf = S_term_frequency(term, document) * idf[term]
            doc_tfidf.append(tfidf)

            doc_tfidf_with_term.append((term, tfidf))
        # Plain
        tfidf = [doc_tfidf, documents[doc_index][0], documents[doc_index][1]]
        tfidf_documents.append(tfidf)

        # with term
        tfidf_with_term = [doc_tfidf_with_term, documents[doc_index][0], documents[doc_index][1]]
        tfidf_documents_with_term.append(tfidf_with_term)
        doc_index += 1
        print("{} from {}".format(doc_index, length_document))

    print("============= TFIDF PLAIN ==============")
    print(tfidf_documents)
    print("")
    print("============= TFIDF WITH TERM ==============")
    print(tfidf_documents_with_term)
    print("")

runner()