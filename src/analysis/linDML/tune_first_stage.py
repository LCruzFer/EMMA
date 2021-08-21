import sys
from pathlib import Path
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor as RFR 
from sklearn.model_selection import GridSearchCV
#* set path to import utils 
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils as du

#* get data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'

'''
In this file tune the hyperparameters of the first stage (Random Forest so far). This is done outside of the estimation file as takes up quite some time and (in theory) only has to be done once. The results are saved in a dataframe and then exported to a CSV so they can be used in the estimation procedure. 
'''

#*#########################
#! FUNCTIONS
#*#########################
def prep_tune_data(df, outcome, treatment, n_test): 
    '''
    Use various utils.data_utils() functions to prepare data for hyperparameter tuning.
    *df=pandas DataFrame containing data 
    *outcome=str with col name of outcome 
    *treatment=str with col name of treatment
    *n_test=int size of test set
    '''
    #split into test and train data 
    y_train, y_test, t_train, t_test, z_train, z_test=du().create_test_train(variables, 'custid', outcome, treatment, n_test=n_test)
    #drop all rebate and expenditure variables that are not treatment or outcome from z_train and z_test
    z_test=du().drop_exp_rbt(z_test)
    z_train=du().drop_exp_rbt(z_train)

    return y_train, y_test, t_train, t_test, z_train, z_test

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
variables=pd.read_csv(data_out/'transformed'/'cleaned_dummies.csv')
#! what to do about all the NaN? 
variables=variables.replace(np.nan, 0)
#choose how many observations in test set
n_test=1000
#choose treatment 
treatment='RBTAMT'
#choose outcome 
outcome='chTOTexp'
#split and prepare 
y_train, y_test, r_train, r_test, z_train, z_test=prep_tune_data(variables, outcome, treatment, 1000)

#*#########################
#! RF TUNING
#*#########################
#tune the two random forests and return best parameters 
#set parameters to look at 
parameters={'max_depth': np.linspace(1, 30, num=5, dtype=int), 'min_samples_leaf': np.linspace(1, 60, num=10, dtype=int), 'max_features': ['auto', 'sqrt', 'log2']}
#rf for Y
best_params_Y=tune_rf(parameters, X=z_train, Y=y_train)
print('Got RF params for Y')
#rf for R 
best_params_R=tune_rf(parameters, X=z_train, Y=r_train)
print('Got RF params for R')
#bind both into dataframe 
params_R_df=pd.DataFrame.from_dict(best_params_R, orient='index').rename(columns={0:'R'})
params_Y_df=pd.DataFrame.from_dict(best_params_Y, orient='index').rename(columns={0:'Y'})
#merge into one 
params_df=params_R_df.merge(params_Y_df, left_index=True, right_index=True)
#write into csv 
params_df.to_csv(data_out/'transformed'/'first_stage_hyperparameters.csv')