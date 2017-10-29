# this use play store twitter corpus
# http://textblob.readthedocs.io/en/dev/classifiers.html#evaluating-classifiers
# http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/
from sklearn.externals import joblib
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *
import random
import collections
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
import itertools
import _pickle as cPickle
import os

# variables
stopset = set(stopwords.words('english')) - {'over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such',
                                             'few', 'so', 'too', 'very', 'just', 'any', 'once'}
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")
stemmer = EnglishStemmer()

def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats

def train():
    with open(path + "/Python/bimbingan_data/twitter_train_23536_1.pickle", "rb") as handle:
        return cPickle.load(handle)

def test():
    with open(path + "/Python/bimbingan_data/twitter_test_15691_1.pickle", "rb") as handle:
        return cPickle.load(handle)

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
                t = token.lower()
                t = stemmer.stem(t)
                if is_string_not_empty(t):
                    valid = t not in set(stopwords.words('english'))
                    if valid:
                        new_sentence += t + " "
        documents.append(new_sentence)
    return documents

def precision_recall(classifier):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    print('pos precision:', classifier.metrics.precision(refsets['pos'], testsets['pos']))
    print('pos recall:', classifier.metrics.recall(refsets['pos'], testsets['pos']))
    print('neg precision:', classifier.metrics.precision(refsets['neg'], testsets['neg']))
    print('neg recall:', classifier.metrics.recall(refsets['neg'], testsets['neg']))


def testing(sentence):
    # cl = joblib.load(path + '/Python/bimbingan_data/sklearn-joblib-train-twitter-1.pkl')
    classifier = NaiveBayesClassifier(train(), feature_extractor=end_word_extractor)
    blob = TextBlob(sentence, classifier=classifier)
    print(sentence + " label : ", blob.classify())
    print("polarity", blob.sentiment.polarity)  # polarity and subjectivity
    print("subjectivity", blob.sentiment.subjectivity)

    ## calc neg and pos
    sentiment = TextBlob(sentence, classifier=classifier, analyzer=NaiveBayesAnalyzer())
    print("positive", sentiment.sentiment.p_pos)
    print("negative", sentiment.sentiment.p_neg)
    # print("Accuracy: {0}".format(classifier.accuracy(test())))

    # test_result = []
    # gold_result = []

    # for i in range(len(test())):
    #    test_result.append(classifier.classify(test()[i][0]))
    #    gold_result.append(test()[i][1])

    # print('Clasification report:\n', classification_report(gold_result, test_result))
    # print('Confussion matrix:\n', confusion_matrix(gold_result, test_result))


def collection():
    datas = []
    datas.append(
        "'It''s a very nice app to use, when it''s working correctly. Been having trouble with it as of late. Trying to get my notifications turned on. It keeps telling me twitter has stopped (Report) (OK). I''ve had to uninstall it and then reinstall it a number of times. If it starts working correctly I''d be very happy to give it a higher rating. Like i mentioned it is a very nice app, when working correctly. After just UNISTALLING the APP & then REINSTALLING it. It seems to be working better. I''ve had to do that about 4 or 5 times so far. ?? '")
    datas.append(
        "'So here is what I hate: when I go to someone''s profile neither on tweets nor on tweets and replies I can't see all of their tweets!! I feel like it''s selected or something but I WANNA SEE ALL TWEETS PLEASE!! Lately I have also noticed that I do not see all the tweets of people I follow on my time line but I WANNA SEE ALL!!! also I wish I could download not only pictures in tweets but also gifs. Oh and sometimes I don''t get any notifications and sometimes I do I don''t know why but it''s annoying, please fix! I love the concept and the fact that it''s the only social media staying itself and not going Facebook or Snapchat with the stories and stuff '")
    datas.append(
        "'Every time they so-called update the app, they add more problems than they solve. Matter how of fact, you don''t even know how they made better. So now, the search button which usually shows trending topics isn''t doing that anymore. And a tweet with a 1000 rt only shows 1 when seen among other tweets. '")
    datas.append(
        "'Trending topics revision sucks with latest update... not as accessible and less robust. Likes and retweets counters are cut off if over 99 so 100 shows as 1 unless you isolate the tweet. Other than that, I''m mostly fine with newer changes. Showing who is replying to whom is helpful but obnoxious. Can''t think of a better option, but it''s unpleasant as is. '")
    return datas


#for doc in collection():
#    testing(sentence=doc)

documents = preprocessing(collection())
for i in range(0, len(documents)):
    print(documents[i])
