import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

df = pd.read_excel('main_dataset_2.xlsx', encoding='ISO-8859-1')
X_data = df[['Phrase']].values
Y_data = df[['Sentiment']].values
# The sentiment labels are:
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive

print(df.head())

processed_features = []

for sentence in range(0, len(X_data)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(X_data[sentence][0]))

    # remove all single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


vectorizer = TfidfVectorizer(max_features=10, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

X_train, X_test, y_train, y_test = train_test_split(processed_features, Y_data, train_size=0.5)
classfier = GaussianNB()
classfier.fit(X_train, y_train)
y_pred = classfier.predict(X_test)
print(accuracy_score(y_test, y_pred))
