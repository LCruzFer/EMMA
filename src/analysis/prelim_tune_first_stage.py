from pathlib import Path
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor as RFR 
from sklearn.model_selection import GridSearchCV
from numpy.random import default_rng

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

'''
In this file tune the hyperparameters of the first stage (Random Forest so far). This is done outside of the estimation file as takes up quite some time and (in theory) only has to be done once. The results are saved in a dataframe and then exported to a CSV so they can be used in the estimation procedure. 
'''

#*#########################
#! FUNCTIONS
#*#########################
def split_data(Y, R, X, W, n_test=2000): 
    '''
    Split data into test and training sample.
    
    #TODO: STREAMLINE FOR DF, SERIES AND ARRAYS
    
    *Y, R, X, W=pandas series and df containing data
    '''
    #sample integers that denote integer of rows used in test set
    #also set seed here
    rng=default_rng(2021)
    rows=rng.integers(0, len(Y)+1, size=n_test)
    #keep rows as test data and rest as train data
    Y_test=Y[rows]
    Y_train=Y[~Y.index.isin(rows)]
    X_test=X.iloc[rows, :]
    X_train=X[~X.index.isin(rows)]
    W_test=W.iloc[rows, :]
    W_train=W[~W.index.isin(rows)]
    R_test=R[rows]
    R_train=R[~R.index.isin(rows)]
    
    return(Y_test, Y_train, R_test, R_train, X_test, X_train, W_test, W_train)

def tune_rf(params, X, Y):
    '''
    Tune a random forest using sklearn's GridSearchCV function using the supplied parameters dictionary and data. Return the optimal parameters.
    
    *params=dictionary of parameters to consider in tuning 
    *X=X training data
    *Y=Y training data
    '''
    #initialize random forest 
    rfr=RFR()
    #apply gridsearch 
    gs_cv=GridSearchCV(rfr, param_grid=params)
    #then fit to data 
    gs_cv.fit(X=X, y=Y)
    #and retrieve best parameters based on this data 
    best=gs_cv.best_params_
    
    return best 

#*#########################
#! DATA
#*#########################
#data right now based on Misra & Surico Dataset
#use observable df generated in prelim_dataset.py
observables=pd.read_csv(data_out/'transformed'/'MS_observables_dummies.csv')
observables=observables.drop('Unnamed: 0', axis=1)
#use df with rebate variables generated in prelim_dataset.py
treatment=pd.read_csv(data_out/'transformed'/'MS_rebate_vars.csv')
treatment=treatment.drop('Unnamed: 0', axis=1)
#use expenditure variables dataset generated in prelim_dataset.py
expenditures=pd.read_csv(data_out/'transformed'/'expenditures.csv')

#*#########################
#! TUNING
#*#########################
'''
First stage in DML consists of two predictions, hence need to tune two RFs. One for outcome Y, one for rebate.
'''
#set variables
#!use same ones as in prelim_dml_runs.py right now
Y=expenditures['chTOTexp']
R=treatment['RBTAMT']
#indicator variable if rebate received as treatment
iR=treatment['iREB']
#for now consider a subset of observables as X; for definitions see MS_columnexplainers.numbers
X=observables[['AGE', 'FAM_SIZE', 'ORGMRTX', 'children', 'FINCBTXM', 'FSALARYM']+['MARITAL1_'+str(num) for num in range(1, 6)]+['SAVA_CTX_'+lit for lit in ['A', 'B', 'C', 'D', 'T']]]
#use rest of observables as W 
W=observables[[col for col in observables.columns if col not in X.columns]]
#!tune rf on training data only 
Y_test, Y_train, R_test, R_train, X_test, X_train, W_test, W_train=split_data(Y, R, X, W, n_test=2000)
#tune the two random forests and return best parameters 
#set parameters to look at 
parameters={'max_depth': np.linspace(0, 30, num=10), 'min_samples_leaf': np.linspace(0, 60, num=20), 'max_features': ['auto', 'sqrt', 'log2']}
#rf for Y
best_params_Y=tune_rf(parameters, X=X_train, Y=Y_train)
print('Got RF params for Y')
#rf for R 
best_params_R=tune_rf(parameters, X=X_train, Y=R_train)
print('Got RF params for R')
#bind both into dataframe 
params_R_df=pd.DataFrame.from_dict(best_params_R, orient='index').rename(columns={0:'R'})
params_Y_df=pd.DataFrame.from_dict(best_params_Y, orient='index').rename(columns={0:'Y'})
#merge into one 
params_df=params_R_df.merge(params_Y_df, left_index=True, right_index=True)
#write into csv 
params_df.to_csv(data_out/'transformed'/'first_stage_hyperparameters.csv')