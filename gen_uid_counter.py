import numpy as np
import pandas as pd
import os
import pickle
from collections import Counter
from preprocess import *

# Load train
cwd = os.getcwd()
train_path = cwd + '/data/train.gz'
save_path = cwd + '/data/uid_counter.pkl'

# Iterating the train in chunks and get counting data
chunksize = 1000000
uid_counter = Counter()
stopper = 0
for c in pd.read_csv(train_path, chunksize=chunksize):
    train = c
    train['uid'] = train['device_ip'] + train['device_model']
    uid_counter += Counter(train['uid'])
    stopper += 1
    if stopper % 5 == 0:
        with open(save_path, 'wb') as f: #save counter once in a while
            pickle.dump(uid_counter, f)
        print('Iteration #', stopper)
        print('Current size of counter: ', len(uid_counter))

# when iteration finished and there are data left
with open(save_path, 'wb') as f: 
    pickle.dump(uid_counter, f)
print('Iteration #', stopper)
print('Current size of counter: ', len(uid_counter))