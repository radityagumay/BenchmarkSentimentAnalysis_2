# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
import random
from nltk.corpus import movie_reviews
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import pymysql
import random

random.seed(1)

conn = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')

# initialize
cur = conn.cursor()
cur.execute("SELECT reviewBody, label FROM google_play_crawler.authors17 where (label = 'pos' or label = 'neg') and length(reviewBody) > 50 limit 0, 100")

def load_train():
    train = []
    for i in cur:
        train.append((i[0], i[1]))
    return train
train = load_train()

# test
cur.execute("SELECT reviewBody, label FROM google_play_crawler.authors17 where (label = 'pos' or label = 'neg') and length(reviewBody) > 50 limit 100, 40")
def load_test():
    test = []
    for i in cur:
        test.append((i[0], i[1]))
    return test
test = load_test()

cl = NaiveBayesClassifier(train)
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]
random.shuffle(reviews)
new_train, new_test = reviews[0:100], reviews[101:200]
cl.update(new_train)

sentence = "Its a very good use of time if you use it to see news"
prob_dist = cl.prob_classify(sentence)
print(prob_dist.max())
print("probability positive:", round(prob_dist.prob("pos"), 2))
print("probability negative:", round(prob_dist.prob("neg"), 2))
print("Accuracy: {0}".format(cl.accuracy(test)))
print("Sentence", cl.classify(sentence))
cl.show_informative_features(5)

cur.close()
conn.close()

