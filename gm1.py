''' To run this program properly, one must first
wget http://konect.uni-koblenz.de/downloads/tsv/moreno_blogs.tar.bz2
tar xfvj moreno_blogs.tar.bz2

pip install -q tensorflow==2.0.0-alpha0
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

FIXED_NEIGHBOR_SIZE = 10
BATCH_SIZE = 32

import random
import csv
from sklearn.model_selection import train_test_split
from collections import Counter
import itertools

def label_to_int(label):
    if label == 'left=leaning':
        return 0
    else:
        return 1

def label_to_vec(label):
    if label == 'left-leaning':
        return [1,0]
    else:
        return [0,1]

def load_data():

    edges = []

    with open('/Users/kellybrower/mystuff/mlg/moreno_blogs/out.moreno_blogs_blogs') as f:
        reader = csv.reader(f, delimiter=' ')
        for i in reader:
            if i[0] != "%":
                edges.append( (int(i[0])-1, int(i[1])-1) )

    labels = []

    with open('/Users/kellybrower/mystuff/mlg/moreno_blogs/ent.moreno_blogs_blogs.blog.orientation') as f:
        reader = csv.reader(f, delimiter=' ')
        for i in reader:
            labels.append(i[0])

    X = []
    y = []

    for (node_id, label) in enumerate(labels):
        neighbors = set()
        for (v1, v2) in edges:
            if v1 == node_id:
                neighbors.add(v2)
            if v2 == node_id:
                neighbors.add(v1)

        try:
            neighbors.remove(node_id)
        except:
            pass

        neighbor_labels = [label_to_vec(labels[i]) for i in neighbors]
        random.shuffle(neighbor_labels)

        X.append(neighbor_labels)
        y.append(label_to_int(label))

    return X, y

X, y = load_data()

print("Data statistics")
print("Number of data-points", len(X))
print("Average number of neighbors", np.average([len(i) for i in X]))
print("Max number of neighbors", np.max([len(i) for i in X]))
print("Min number of neighbors", np.min([len(i) for i in X]))
print("Distribution of labels", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

print()
print("Number of training examples", len(X_train))
print("Distribution of labels", Counter(y_train))
print("Example: ", X_train[90], " has label ", y_train[90])

print()
print("Number of test examples", len(X_test))
print("Distribution of labels", Counter(y_test))
print("Example: ", X_test[0], " has label ", y_test[0])
