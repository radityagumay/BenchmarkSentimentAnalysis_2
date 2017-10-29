from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import sys, unicodedata, os
import nltk
nltk.download('punkt')

stemmer = PorterStemmer()
tablePunctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                                 if unicodedata.category(chr(i)).startswith('P'))

document = "Twitter Its descent Im on it a lot more now but 1 thing it " \
           "needs is a way to chat live with people. I think it would be " \
           "cool to enhance the bubble tweets so people can live chat on " \
           "Twitter with the followers"

def preprocessing():
    sentence = ""
    tokens = word_tokenize(document)
    for token in tokens:
        if len(token) > 3:
            stem = stemmer.stem(token)
            punct = stem.translate(tablePunctuation)
            if punct is not None:
                stop = punct not in set(stopwords.words('english'))
                if stop:
                    sentence += str(punct.lower()) + " "
    return sentence

doc = preprocessing()

print(document, len(word_tokenize(document)))
print(doc, len(word_tokenize(doc)))

