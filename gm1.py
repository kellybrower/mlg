''' 
To run this program properly, one must first
wget http://konect.uni-koblenz.de/downloads/tsv/moreno_blogs.tar.bz2
tar xfvj moreno_blogs.tar.bz2
Make sure your tensorflow and scikit are up to date: 
pip3 install -q tensorflow==2.0.0-alpha0
pip3 install -U scikit-learn
(pip3 since we are using python3)
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
    if label == 'left-leaning':
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)

print()
print("Number of training examples", len(X_train))
print("Distribution of labels", Counter(y_train))
print("Example: ", X_train[90], " has label ", y_train[90])

print()
print("Number of test examples", len(X_test))
print("Distribution of labels", Counter(y_test))
print("Example: ", X_test[0], " has label ", y_test[0])

def fix_size_of_list(data, target_len=FIXED_NEIGHBOR_SIZE):
  '''
  This function highlights one of the central challenges of graph data:
  it is naturally variable sized and frameworks like TensorFlow want
  fixed sized tensor data.
  
  Our simplistic solution is to fix the size - we chop it down if too large, or
  zero pad it if too small.
  '''
  
  delta = len(data) - target_len
  
  if delta >= 0:
    return data[0:target_len]
  else:
    return np.pad(data, [(0, -delta), (0,0)], mode='constant', constant_values=0)


# Create TensorFlow dataset objects ready for training and evaluation

## Training data

X_train_fixed = [fix_size_of_list(i) for i in X_train]

dataset_train = tf.data.Dataset.from_tensor_slices(( X_train_fixed , y_train))
dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_train = dataset_train.shuffle(BATCH_SIZE * 10)

## Test data

X_test_fixed = [fix_size_of_list(i) for i in X_test]

dataset_test = tf.data.Dataset.from_tensor_slices(( X_test_fixed , y_test))
dataset_test = dataset_test.batch(BATCH_SIZE)

model = keras.Sequential([
  layers.Input(shape=[FIXED_NEIGHBOR_SIZE, 2]),
  layers.Flatten(),
  layers.Softmax()
])
model.add(layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training:")
model.fit(dataset_train, epochs=13, verbose=1)

print("\n\nFinal test accuracy:")

results = model.evaluate(dataset_test)

for l, v in zip(model.metrics_names, results):
  print(l, v)

