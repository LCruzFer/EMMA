import pandas as pd
from sklearn.model_selection import train_test_split 

'''
This file contains various helper functions for that are done in several files but always involve the same steps - e.g. splitting data into train and test sets or other things.
'''

def create_test_train(data, outcome, treatment, n_test=2000):
    '''
    Create single datasets for outcome, treatment and controls - train and test sets respectively.
    *data=df with all data 
    *outcome=str that is column name of outcome
    *treatment=str that is column name of treatment
    '''
    train, test=train_test_split(data, test_size=n_test, random_state=2021)
    y_train=train[outcome]
    y_test=test[outcome]
    t_train=train[treatment]
    t_test=test[treatment]
    z_train=train[[col for col in train.columns if col not in [outcome, treatment]]]
    z_test=test[[col for col in train.columns if col not in [outcome, treatment]]]
    return y_train, y_test, t_train, t_test, z_train, z_test

def drop_exp_rbt(df):
    '''
    Drop all columns related to expenditures or rebate that are in df. 
    *df=pandas df
    '''
    df=df[[col for col in df.columns if 'exp' not in col]]
    df=df[[col for col in df.columns if 'rbt' not in col]]
    df=df[[col for col in df.columns if 'reb' not in col]]
    df=df[[col for col in df.columns if 'REB' not in col]]
    return df

def split_XW(Z, x_columns):
    '''
    Split all observables into X and W data.
    *Z=all observables 
    *x_columns=columns that are supposed to be in X
    '''
    #get x data
    x=Z[x_columns]
    #then get w data 
    w=Z[[col for col in Z.columns if col not in x.columns]]
    return (x, w)