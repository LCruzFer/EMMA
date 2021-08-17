from pathlib import Path
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor as RFR 
from sklearn.model_selection import GridSearchCV
import utils

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
#read in demeaned data
demeaned_vars=pd.read_csv(data_out/'transformed'/'prepped_data_demeaned.csv')
#choose outcome 
outcome='chTOTexp'
#choose treatment
treatment='RBTAMT'
#split into test and train data and then create subset dataframes
y_train, y_test, r_train, r_test, z_train, z_test=utils.create_test_train(demeaned_vars, outcome, treatment)
#drop all other expenditure and rebate variables from Z data 
z_train=utils.drop_exp_rbt(z_train)
z_test=utils.drop_exp_rbt(z_test)

#some columns have lots of NaN Z - drop for now and 
#!INVESTIGATE LATER ON 
z_train=z_train.drop(['QESCROWX', 'QBLNCM1X', 'ORGMRTX', 'nmort', 'timeleft', 'NEWMRRT', 'QMRTTERM', 'payment'], axis=1) 
z_test=z_test.drop(['QESCROWX', 'QBLNCM1X', 'ORGMRTX', 'nmort', 'timeleft', 'NEWMRRT', 'QMRTTERM', 'payment'], axis=1)

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