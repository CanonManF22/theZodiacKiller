import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
import time
from sklearn.feature_extraction.text import TfidfVectorizer

trainDF = pd.read_csv('Dataset.csv')
print(trainDF.dtypes)

trainDF = pd.read_csv('Dataset.csv')
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}',ngram_range=(2,7), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(trainDF['text'])

xtrain_tfidf_ngram_chars.shape

train_x,test_x,train_y,test_y = train_test_split(xtrain_tfidf_ngram_chars,trainDF['Label'],test_size=0.3)

linear_svc = LinearSVC()
start_time = time.time()
linear_svc.fit(train_x, train_y)
print('Score:', linear_svc.score(test_x, test_y))
