import numpy as np
import pandas as pd
import os
import pickle

from preprocess import *

mode = 'debug' #mode options: debug, 1, n, full 
# if it's int - then it means n_portions

# Load counter
print('Loading counter...')
cwd = os.getcwd() # current dir
uid_counter = cwd + '/data/uid_counter.pkl'

with open(uid_counter, 'rb') as f:
    uid_counter = pickle.load(f)

print('  Done. Show a few items in the counter: ', uid_counter.most_common(3))

# Training 
print('Training mode: ', mode)
train_path = cwd + '/data/train.gz'

if mode == 'debug':
    chunksize = 1000
    for eid,c in enumerate(pd.read_csv(train_path, chunksize=chunksize)):
        train = c
        print('The shape of training dataset: ', train.shape)
        encoded_data, target, encoder = encoding(train, uid_counter)
        clf, loss = training(encoded_data, target)
        save_model(clf, encoder, loss, cwd, model_index=eid)
        break

elif type(mode) == int:
    print(f'Training {mode} portions')
    chunksize = 4000000
    for eid,c in enumerate(pd.read_csv(train_path, chunksize=chunksize)):
        print('Training portion number ', eid)
        train = c
        print('The shape of training dataset: ', train.shape)
        encoded_data, target, encoder = encoding(train, uid_counter)
        clf, loss = training(encoded_data, target)
        save_model(clf, encoder, loss, cwd, model_index=eid)
        print('')
        if eid == mode:
            break