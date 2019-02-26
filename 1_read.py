import numpy as np
import csv

train_arr = []
with open("mnist_train.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        train_arr.append([int(x) for x in line])
train_arr = np.array(train_arr)
train_labels = train_arr[:,0]
train_arr = train_arr[:,1:]

test_arr = []
with open("mnist_test.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        test_arr.append([int(x) for x in line])
test_arr = np.array(test_arr)
test_labels = test_arr[:,0]
test_arr = test_arr[:,1:]

