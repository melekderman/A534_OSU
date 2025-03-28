#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
import pandas as pd
from svector import svector
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter


def read_data(textfile):
    data = pd.read_csv(textfile)
    data['target'] = data['target'].apply(lambda x: 1 if x == '+' else 0)
    return data

def preprocess_data(data):
    word_counts = Counter()
    for words in data['sentence']:
        word_counts.update(words.split())

    common_words = {'the'}
    filtered_data = data['sentence'].apply(lambda x: ' '.join(
        [word for word in x.split() if word_counts[word] > 1 and word not in common_words]))
    data['sentence'] = filtered_data
    return data

def make_vector(words):
    v = svector()
    v['<bias>'] = 1
    for word in words.split():
        v[word] += 1
    return v

def vectorize_data(data, all_keys=None):
    vectors = data['sentence'].apply(make_vector).tolist()
    y = data['target'].values

    if all_keys is None:
        all_keys = list(set(key for vector in vectors for key in vector.keys()))

    X = np.zeros((len(vectors), len(all_keys)))
    key_index = {key: idx for idx, key in enumerate(all_keys)}

    for i, vector in enumerate(vectors):
        for key, value in vector.items():
            if key in key_index:
                X[i, key_index[key]] = value
                
    return X, y, all_keys

def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def train_and_evaluate(trainfile, devfile):
    t = time.time()
    train_data = read_data(trainfile)
    dev_data = read_data(devfile)
    
    train_data = preprocess_data(train_data)
    dev_data = preprocess_data(dev_data)
    
    X_train, y_train, all_keys = vectorize_data(train_data)
    X_dev, y_dev, _ = vectorize_data(dev_data, all_keys)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    model = SVC()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_dev_scaled)
                
    accuracy = accuracy_score(y_dev, y_pred)
    print(f"SVC Accuracy: {accuracy:.4f}")
        
    return model, scaler, all_keys

def blind_test(testfile, model, scaler, all_keys):
    data = pd.read_csv(testfile)
    data = preprocess_data(data)
    X_test, _, _ = vectorize_data(data, all_keys)
    X_test_scaled = scaler.transform(X_test)
    
    predictions = model.predict(X_test_scaled)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        if label == "?":
            prediction = model.dot(make_vector(words.split()))
            data.at[i, 'target'] = '+' if prediction > 0 else '-'
            predictions.append(data.iloc[i])
            
    output_file = "svc_predictions.csv"
    data['prediction'] = predictions
    data.to_csv(output_file, index=False)
    print(f"SVC predictions saved to {output_file}")
    
    return data

if __name__ == "__main__":
    trainfile = sys.argv[1]
    devfile = sys.argv[2]
    testfile = sys.argv[3]
    test_updated = sys.argv[4]
    epochs = 10
    
    model, scaler, all_keys = train_and_evaluate(trainfile, devfile)
    updated_test_data = blind_test(testfile, model, scaler, all_keys)
