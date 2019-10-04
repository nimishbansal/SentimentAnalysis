import os
import time
from string import punctuation

import nltk
import pandas as pd
# from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
nltk.download('punkt')
os.listdir(".")

stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', punctuation)


# print(stop_words)
# print(punctuation)
# print(table)

def textclean(text):
    tokens = word_tokenize(text)
    #     print(tokens)
    tokens = [word for word in tokens if word.isalpha()]
    #     print(tokens)
    tokens = [w.translate(table) for w in tokens]
    #     print(tokens)
    tokens = [word for word in tokens if word not in stop_words]
    #     print(tokens)
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


sentence = "Line that shows when sentence is converted to list of words. Isn't it cool"
# print(textclean(sentence))


sentiment_dictionary = {0: 'negative', 2: 'neutral', 4: 'positive'}

df = pd.read_excel('dataset3.xlsx').reset_index(drop=True).iloc[:1000]

df = df[[0, 5]]
df.columns = ['label', 'tweet']
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[['tweet']], df[['label']])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

tweets = []
for i in range(len(X_train)):
    words = X_train.iloc[i]['tweet']
    sentiment = y_train.iloc[i]['label']
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment_dictionary[sentiment]))

for current_tweet, sentiment in tweets[:5]:
    print(current_tweet, sentiment)


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


word_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets)

a = time.time()
classifier = nltk.classify.SklearnClassifier(DecisionTreeClassifier())

classifier.train(training_set)

print(time.time() - a)

a = time.time()
predicted = classifier.classify_many([extract_features(tweet.split()) for tweet in X_test['tweet']])

y_pred = []
for fr in predicted:
    if fr == 'negative':
        y_pred.append(0)
    else:
        y_pred.append(4)

y_true = list(y_test.values)
print("accuracy is ",accuracy_score(y_true, y_pred))
print(time.time() - a)
