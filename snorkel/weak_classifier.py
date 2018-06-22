from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

X_train = []
Y_train = []

vectorizer = TfidfVectorizer(min_df = 2, max_df = 0.8)

def load_data():
    FPATH = 'data/annotated_tweets.tsv'
    annotated_tweets = pd.read_csv(FPATH, sep='\t')
    for index, row in annotated_tweets.iterrows():
        X_train.append(row['content'])
        Y_train.append(row['label'])

def train_classifier():
    load_data()

    X_train_vecs = vectorizer.fit_transform(X_train)
    clf = LogisticRegression()
    clf.fit(X_train_vecs, Y_train)

    return vectorizer, clf

def classify(vectorizer, clf, features):
    '''Vectorizes the text, loads the trained classifier and classifies the text in question.'''

    vec_features = vectorizer.transform(features)
    return clf.predict(vec_features)[0]
