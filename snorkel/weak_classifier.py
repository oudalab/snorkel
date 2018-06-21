from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import StableLabel

X_train = []
Y_train = []

vectorizer = TfidfVectorizer(min_df = 2, max_df = 0.8)

def load_data(session, annotator_name='gold'):
    res_set = session.query(StableLabel).filter(StableLabel.annotator_name == annotator_name)
    for res in res_set:
        X_train.append(res.tweet)
        Y_train.append(res.value)

def train_classifier(session, annotator_name):
    '''Trains classifier and saves it using pickle.'''

    load_data(session, annotator_name)

    X_train_vecs = vectorizer.fit_transform(X_train)
    clf = LogisticRegression()
    clf.fit(X_train_vecs, Y_train)

    return vectorizer, clf

def classify(vectorizer, clf, features):
    '''Vectorizes the text, loads the trained classifier and classifies the text in question.'''

    vec_features = vectorizer.transform(features)
    return clf.predict(vec_features)[0]
