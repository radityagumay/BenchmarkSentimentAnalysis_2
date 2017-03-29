# http://stackoverflow.com/a/16344128/5435658

from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize
from itertools import chain
import csv
import json
import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()

cur.execute("SELECT reviewBody,label FROM review_label_benchmark_with_polarity limit 2")

training_data = []

for r in cur:
    print(r)
    training_data.append(r)

cur.close()
conn.close()

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))

feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in
               training_data]


def save_to_file():
    with open('feature.json', 'w') as outfile:
        json.dump(feature_set, outfile)


save_to_file()


def load_feature():
    with open('feature.json', 'r') as infile:
        print(json.load(infile))


load_feature()

classifier = nbc.train(feature_set)

test_sentence = "Twitter Great & Fun app to have!!"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}

print("test_sent:", test_sentence)
print("tag:", classifier.classify(featurized_test_sentence))

cur.close()
conn.close()

# training_data = [('I love this sandwich.', 'pos'),
#                  ('This is an amazing place!', 'pos'),
#                  ('I feel very good about these beers.', 'pos'),
#                  ('This is my best work.', 'pos'),
#                  ("What an awesome view", 'pos'),
#                  ('I do not like this restaurant', 'neg'),
#                  ('I am tired of this stuff.', 'neg'),
#                  ("I can't deal with this", 'neg'),
#                  ('He is my sworn enemy!', 'neg'),
#                  ('My boss is horrible.', 'neg')]
#
# vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
#
# feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in
#                training_data]
#
# classifier = nbc.train(feature_set)
#
# test_sentence = "This is the best band I've ever heard!"
# featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}
#
# print("test_sent:", test_sentence)
# print("tag:", classifier.classify(featurized_test_sentence))
