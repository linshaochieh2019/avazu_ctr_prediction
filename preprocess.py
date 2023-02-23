import numpy as np
import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix,hstack
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler,MinMaxScaler,FunctionTransformer,OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def hour2cosine(hr):
    cosine = np.cos(2 * np.pi * hr / 24)
    sine = np.sin(2 * np.pi * hr / 24)
    return cosine,sine

def preprocess_hour(df):
    df['hour'] = df['hour'] % 100
    df['hr0'] = df['hour'].apply(lambda x: hour2cosine(x)[0])
    df['hr1'] = df['hour'].apply(lambda x: hour2cosine(x)[1])
    df = df.drop(columns='hour')
    return df

def encode_site_and_app(df):
    # from site/app to prod
    prod_id = list()
    prod_domain = list()
    prod_category = list()
    for i,row in df.iterrows():
        if row['site_id'] == '85f751fd':
            prod_id.append(row['app_id'])
            prod_domain.append(row['app_domain'])
            prod_category.append(row['app_category'])
        else:
            prod_id.append(row['site_id'])
            prod_domain.append(row['site_domain'])
            prod_category.append(row['site_category'])

    # add production features
    df['prod_id'] = prod_id
    df['prod_domain'] = prod_domain
    df['prod_category'] = prod_category

    # delete unwated columns
    unwanted = ['app_id', 'app_domain', 'app_category', 'site_id', 'site_domain', 'site_category']
    df = df.drop(columns=unwanted)
    return df

def df2matrix_onehot(df, encoder=None):
    num_cols = ['hr0', 'hr1']
    num_data = csr_matrix(df[num_cols])
    df = df.drop(columns=num_cols) #temporailty remove numerical columns so we can do one-hot encoding

    if encoder: #with given encoder
        encoded_data = encoder.transform(df)
        encoded_data = hstack([encoded_data, num_data]) #add numerical columns back

    else: #without given encoder
        encoder = OneHotEncoder(handle_unknown='ignore') #in case there are data that only exists in the train
        encoded_data = encoder.fit_transform(df)
        encoded_data = hstack([encoded_data, num_data]) #add numerical columns back

    return encoded_data,encoder

def gen_uid_count(df, uid_counter):

    temp = pd.DataFrame(
        index = df.index,
        columns = ['uid']
    )

    temp['uid'] = df['device_ip'] + df['device_model']
    temp['uid_cnt'] = temp['uid'].map(uid_counter)

    # Take log for normalization
    log_transformer = FunctionTransformer(np.log)
    uid_cnts = np.array(temp['uid_cnt']).reshape(-1,1) + 1e-2 # added epsilon otherwise cannot take log
    uid_cnts = log_transformer.fit_transform(uid_cnts)

    return uid_cnts

def encoding(df, uid_counter, encoder=None):
    print('Encoding training dataset then return encoded data and encoder...')
    try: 
        target = np.array(df['click'])
        df = df.drop(columns='click')

    except: #in case there is no 'click' data when encoding test
        target = None
        
    # preprocess hour
    print('- encoding hour')
    df['hour'] = df['hour'] % 100
    df['hr0'] = df['hour'].apply(lambda x: hour2cosine(x)[0])
    df['hr1'] = df['hour'].apply(lambda x: hour2cosine(x)[1])
    df = df.drop(columns='hour')

    # preprocess site, app to prod
    print('- encoding site and app data')
    df = encode_site_and_app(df)

    # get counting features
    print('- mapping counting data')
    uid_cnts = gen_uid_count(df, uid_counter)

    # drop unwanted columns
    unwanted = ['id', 'device_id', 'device_ip', 'device_model']
    df = df.drop(columns=unwanted)

    # encoding
    print('- one-hot encoding')
    if not encoder: #training and generate new encoder
        encoded_data,encoder = df2matrix_onehot(df, encoder=None) 
    else: # encoding data using given encoder
        print('    - using given encoder')
        encoded_data,encoder = df2matrix_onehot(df, encoder=encoder)
         
    # add counting features back to encoded data
    encoded_data = hstack([encoded_data, uid_cnts])
    print('Encoding is done')
    return encoded_data, target, encoder

def training(encoded_data, target):
    print('Training model')
    X_train, X_val, y_train, y_val = train_test_split(encoded_data, target, test_size=0.1)
    clf = XGBClassifier(objective='binary:logistic')
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return clf, loss

def save_model(clf, encoder, loss, cwd, model_index):
    print('Saveing model, encoder and loss')
    model = {
        'clf': clf,
        'encoder': encoder,
        'loss': loss
    }
    model_name = cwd + f'/clfs/clf_{model_index}.pkl'

    # check clfs dir exists otherwise mkdir
    try:
        clfs_path = cwd + '/clfs'
        os.mkdir(clfs_path)
    
    except FileExistsError:
        pass #"Directory already exists"


    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
    print('Model and encoder saved')