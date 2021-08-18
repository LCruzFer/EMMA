import pandas as pd
from sklearn.model_selection import train_test_split 

'''
This file contains various helper functions for that are done in several files but always involve the same steps - e.g. splitting data into train and test sets or other things.
'''
class utils: 
    '''
    A class containing tools for data pre-processing and manipulation.
    '''
    def __init__(self, df): 
        self.df=df

    def get_expnames(self): 
        '''
        Filter self.df for all expenditure variables.
        *self.df=pandas dataframe, expenditure variables signaled by 'exp' in column name
        '''
        #get columns that have 'exp' in their name
        exp_cols=[col for col in self.df.columns if 'exp' in col]
        #also filter between level, change and lag 
        lvl=[col for col in exp_cols if ('ch' not in col) & ('last' not in col)]
        ch=[col for col in exp_cols if 'ch' in col]
        lag=[col for col in exp_cols if 'last' in col]
        return lvl, ch, lag

    def drop_exp_rbt(self):
        '''
        Drop all columns related to expenditures or rebate that are in self.df. 
        *self.df=pandas self.df
        '''
        df=self.df[[col for col in self.df.columns if 'exp' not in col]]
        df=self.df[[col for col in self.df.columns if 'rbt' not in col]]
        df=self.df[[col for col in self.df.columns if 'reb' not in col]]
        df=self.df[[col for col in self.df.columns if 'REB' not in col]]
        df=self.df[[col for col in self.df.columns if 'RBT' not in col]]
        return df

    
    def create_test_train(self, outcome, treatment, n_test=2000):
        '''
        Create single datasets for outcome, treatment and controls - train and test sets respectively.
        *data=self.df with all data 
        *outcome=str that is column name of outcome
        *treatment=str that is column name of treatment
        '''
        #!change this to split based on id, such that households are either in training or test set, not in both
        train, test=train_test_split(self.df, test_size=n_test, random_state=2021)
        y_train=train[outcome]
        y_test=test[outcome]
        t_train=train[treatment]
        t_test=test[treatment]
        z_train=self.drop_exp_rbt(train[[col for col in train.columns if col not in [outcome, treatment]]])
        z_test=self.drop_exp_rbt(test[[col for col in train.columns if col not in [outcome, treatment]]])
        return y_train, y_test, t_train, t_test, z_train, z_test

    def split_XW(self, Z, x_columns):
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
