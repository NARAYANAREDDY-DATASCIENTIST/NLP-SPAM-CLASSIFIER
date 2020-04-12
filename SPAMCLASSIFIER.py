# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:18:39 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""
# IMPORT LIBRARIES

import pandas as pd

# READ THE DATASET

messages=pd.read_csv('SMSSpamcollection.txt',sep='\t',names=["label","message"])

# DATA CLEANING AND PREPROCESSING

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus=[]
for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]', ' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
# CREATING BAG OFWORDS MODEL
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
x=cv.fit_transform(corpus).toarray()

# ENCODING THE CATEGORICAL DATA
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# SPLIT THE TRAIN AND TEST DATA

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# BUILD THE NAIVEBAYES MODL

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB()

# FIT THE MODEL
spam_detect_model.fit(x_train,y_train)

# PREDICT THE MODEL
y_pred=spam_detect_model.predict(x_test)

# PERFORMACE METRICS FOR MODEL
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion_matrix=confusion_matrix(y_test,y_pred)
accuracy_score=accuracy_score(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)