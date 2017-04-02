from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import _pickle as cPickle
import sys, unicodedata

stemmer = EnglishStemmer()

# Load unicode punctuation
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))


# load our local model vocabulary
def load_model_vocabulary():
    with open('model_vocabulary.pickle', 'rb') as handle:
        return cPickle.load(handle)


print(load_model_vocabulary())

def load_feature():
    with open('features.pickle', 'rb') as handle:
        return cPickle.load(handle)


def is_string_empty(string):
    if string == "":
        return False
    return True


classifier = nbc.train(load_feature())

test_sentence = "Twitter the best app ever i use"

def preprocessing_testing_data():
    sentence = ""
    sentence_tokenized = word_tokenize(test_sentence)
    for i in sentence_tokenized:
        lower = i.lower()  # toLower().Case
        if len(lower) > 3:  # only len > 3
            stem = stemmer.stem(lower)  # root word
            punc = stem.translate(tbl)  # remove ? ! @ etc
            if is_string_empty(punc):  # check if not empty
                stop = punc not in set(stopwords.words('english'))
                if stop:  # only true we append
                    sentence += str(punc) + " "
    return sentence


clean_sentence = preprocessing_testing_data()
featurized_test_sentence = {i: (i in word_tokenize(clean_sentence)) for i in load_model_vocabulary()}

print("test_sentence:", clean_sentence)
print("label:", classifier.classify(featurized_test_sentence))
