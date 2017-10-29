# -*- coding: utf-8 -*-

import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import math

stopwords = nltk.corpus.stopwords.words('portuguese')
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)


def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
                    float(num_docs_containing(word, list_of_docs)))


def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))


vocabulary = []
docs = {}
all_tips = []
for tip in (['documment 1', 'documment 2']):
    tokens = tokenizer.tokenize(tip)

    bi_tokens = bigrams(tokens)
    tri_tokens = trigrams(tokens)

    tokens = [token.lower() for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in stopwords]

    print("tokens", tokens)

    bi_tokens = [' '.join(token).lower() for token in bi_tokens]
    bi_tokens = [token for token in bi_tokens if token not in stopwords]

    print("bi tokens", bi_tokens)

    tri_tokens = [' '.join(token).lower() for token in tri_tokens]
    tri_tokens = [token for token in tri_tokens if token not in stopwords]

    print("tri tokens", tri_tokens)

    final_tokens = []
    final_tokens.extend(tokens)
    final_tokens.extend(bi_tokens)
    final_tokens.extend(tri_tokens)
    docs[tip] = {'freq': {}, 'tf': {}, 'idf': {},
                 'tf-idf': {}, 'tokens': []}
    for token in final_tokens:
        # The frequency computed for each tip
        docs[tip]['freq'][token] = freq(token, final_tokens)
        # The term-frequency (Normalized Frequency)
        docs[tip]['tf'][token] = tf(token, final_tokens)
        docs[tip]['tokens'] = final_tokens

    vocabulary.append(final_tokens)

for doc in docs:
    for token in docs[doc]['tf']:
        # The Inverse-Document-Frequency
        docs[doc]['idf'][token] = idf(token, vocabulary)
        # The tf-idf
        docs[doc]['tf-idf'][token] = tf_idf(token, docs[doc]['tokens'], vocabulary)

# Now let's find out the most relevant words by tf-idf.
words = {}
for doc in docs:
    for token in docs[doc]['tf-idf']:
        if token not in words:
            words[token] = docs[doc]['tf-idf'][token]
        else:
            if docs[doc]['tf-idf'][token] > words[token]:
                words[token] = docs[doc]['tf-idf'][token]

    print(doc)
    for token in docs[doc]['tf-idf']:
        print(token, docs[doc]['tf-idf'][token])

for item in sorted(words.items(), key=lambda x: x[1], reverse=True):
    print("%f <= %s" % (item[1], item[0]))

class BigramTfIdf:
    def documents(self):
        document_0 = "Force close Once I updated my header it just automatically closing down and I cant even look at my wall cause Twitter isnt responding but I can tweet anyway please fix. Thanks"
        document_1 = "Draft button media viewing style! 1)PUT THE DRAFT BUTTON BACK AS AN OPTION ON THE 3DOTS AT THE TL PAGE!!! 2)FIX THE STYLE THE MEDIA IS SEEN! MULTIPLE IN A ROW NOT SCROLL DOWN FOR MORE!??????? *DESIGN WISE EACH UPDATE GETS WORSE!!!!"
        document_2 = "Ok One less star because muted users still appear on my lists when they are re-tweeted and the fact that I cant seem to be able to mute hashtags. Another star removed because you either dont care about your users or are too lazy to update the Whats new section."
        document_3 = "Priorities If you follow a lot of pages and friends its very hard to notice all of your friends tweets. I think Twitter should prioritize tweets from friends then only pages. Only through this you can catch up on what you missed. Or a priority settings... Notifications for tweets from starred accounts sometimes dont even pop up at all. Please fix this issue as soon as possible."
        document_4 = "Hearts... Not really When I hit the heart button the app crashes immediately its been like that ever since the introduction of it. Maybe its time to finally do something? After MONTHS?"
        document_5 = "Poor system on phone verification My first account has been locked and so i followed your instruction but i could not proceed to the next step since ive been receiving the code from u super late and when i entered the code the page already expired(found error 404). I then decided to create a new account since yesterday but still unable to sign it up until now. Been posting my cp no. twice for the phone verification section but couldnt receive any code until now.As per your articlewell wait the code for just few minutes.pls fix it."
        document_6 = "I cannot see any videos The app closes or it brings me back to my TL if Im viewing someones profile. "
        document_7 = "You guys suck thats all. Too much useless and frequent updates. Please keep the app stable first before releasing it to the public. Plus please update the version notes on the whats new? section. Hate to see the weve add comments bla bla bla. Please be serious about this guys."
        document_8 = "No change after many updates Followers are still not in any kind of order... for months now... new followers are in the middle and old followers are at the top... not just me but anyone I follow. Also videos play very sluggish if at all "
        document_9 = "tweets take forever to send Have to say not surprising the amount of updates that seem to to nothingOne day Twitter will do an updateand actually make it betterSadly probably wont be in our lifetimes. jumps so u have to scroll bk down to try find tweet FFS how long have people complaind about this and still it does it."
        return [[document_0, "bug"], [document_1, "ui"], [document_2, "feature"], [document_3, "ui"],
                [document_4, "bug"], [document_5, "bug"], [document_6, "bug"], [document_7, "feature"],
                [document_8, "feature"], [document_9, "feature"]]

    def clean_documents(self):
        documents = self.documents()
        new_documents = []
        for document in documents:
            new_documents.append(document[0])

        return new_documents

    def preprocessing(self):
        documents = self.clean_documents()

        tokenized_document = []
        for document in documents:
            tokens = tokenizer.tokenize(document)
            bi_tokens = bigrams(tokens)
            bi_tokens = [' '.join(token).lower() for token in bi_tokens]
            bi_tokens = [token for token in bi_tokens if token not in stopwords]
            tokenized_document.append(bi_tokens)

        for document in tokenized_document:
            print(document)

code = BigramTfIdf()
code.preprocessing()






