#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:44:22 2018

@author: pavan
"""

from flask import Flask
from flask import render_template, request

# modeling packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib

# build a basic model for titanic survival
titanic_df = pd.read_csv('data/titanic_data.csv')
titanic_df['sex_binary'] = titanic_df['sex'].map({'female': 1, 'male': 0})

# choose our features and create test and train sets
features = [u'pclass', u'age', u'sibsp', u'parch', u'fare', u'sex_binary', 'survived']
titanic_df = titanic_df[features].dropna()

features.remove('survived')
X_train = titanic_df[features]
y_train = titanic_df['survived']

# fit the model
L1_logistic = LogisticRegression(C=1.0, penalty='l1')
L1_logistic.fit(X_train, y_train)

joblib.dump(L1_logistic,'model.pkl')

