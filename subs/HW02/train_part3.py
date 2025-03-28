#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
import pandas as pd
from svector import svector
from collections import Counter


def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    v['<bias>'] = 1
    for word in words:
        v[word] += 1
    return v
    
def counter(trainfile):
    word_count = Counter()
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        word_count.update(words)
    return word_count

    
def make_vector_word_filter(words, word_count):
    v = svector()
    v['<bias>'] = 1
    for word in words:
        if word_count[word] > 1:
            v[word] += 1
    return v
    
    
def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def train(trainfile, devfile, epochs=5):
    word_count = counter(trainfile)
    t = time.time()
    best_err = 1.
    W = svector()
    W_a = svector()
    model = svector()
    c = 0
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector_word_filter(words, word_count)
            if label * (W.dot(sent)) <= 0:
                updates += 1
                W += label * sent
                W_a += c * label * sent
                model = (c * W) - W_a
            c += 1
        dev_err = test(devfile, model)
        best_err = min(best_err, dev_err)
        print("epoch %d, c %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return model

def blind_test(testfile, model, output_file="updated_test.csv"):
    data = pd.read_csv(testfile)
    predictions = []
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        if label == "?":
            prediction = model.dot(make_vector(words.split()))
            data.at[i, 'target'] = '+' if prediction > 0 else '-'
            predictions.append(data.iloc[i])
    return data

    
if __name__ == "__main__":
    trainfile = sys.argv[1]
    devfile = sys.argv[2]
    testfile = sys.argv[3]
    test_updated = sys.argv[4]
    epochs = 10
    
    model = train(trainfile, devfile, epochs)
    updated_test_data = blind_test(testfile, model)
    

    updated_test_data.to_csv(test_updated, index=False)
    print(f"Updated test file saved as {test_updated}")