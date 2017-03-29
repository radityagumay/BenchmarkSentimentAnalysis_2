from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize
from itertools import chain

import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

cur = conn.cursor()

cur.execute("SELECT reviewBody,label FROM review_label_benchmark_with_polarity limit 50")

training_data = []

for r in cur:
    print(r)
    training_data.append(r)

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))

feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in
               training_data]

file = open("feature.txt", "w")
file.write(feature_set)

local_feature = open("feature.txt", "r")

classifier = nbc.train(file.read())

test_sentence = "Twitter Great & Fun app to have!!"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}

print("test_sent:", test_sentence)
print("tag:", classifier.classify(featurized_test_sentence))

file.close()
local_feature.close()
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
