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
#load data
variables=pd.read_csv(data_out/'transformed'/'prepped_data.csv')
#set treatment
treatment='RBTAMT'
#set outcomes
outcomes=['chTOTexp', 'chNDexp', 'chSNDexp','chFDexp','chUTILexp', 'chVEHINSexp', 'chVEHFINexp']
#constants 
constants=['const'+str(i) for i in range(1, 15)]

#*#########################
#! RF TUNING
#*#########################
#set parameters to look at 
parameters={'max_depth': np.linspace(1, 30, num=10, dtype=int), 'min_samples_leaf': np.linspace(1, 60, num=20, dtype=int), 'max_features': ['auto', 'sqrt', 'log2']}
#set observables for each specification 
spec1=['AGE', 'AGE_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']+constants
spec2=spec1+['married']
spec3=spec2+['liqassii', 'FINCBTXM', 'FSALARYM']
spec4=spec3+['ORGMRTX', 'owned_m', 'notowned', 'QBLNCM1X']
specs=[spec1, spec2, spec3, spec4]
#set up dataframe on which results will be merged 
all_params=pd.DataFrame(index=['max_depth', 'min_samples_leaf', 'max_features'])
#for each specification, tune random forest for treatment and for each outcome variables
for i, spec in enumerate(specs):
    #select data 
    spec_df=variables[spec+outcomes+[treatment]]
    spec_df=spec_df.dropna()
    #get treatment data 
    treat=spec_df[treatment]
    #tune forest for treatment
    best_params_R=tune_rf(parameters, spec_df[spec], treat)
    #turn into df 
    params_R_df=pd.DataFrame.from_dict(best_params_R, orient='index').rename(columns={0: treatment+'_spec'+str(i+1)})
    #merge onto final params df 
    all_params=all_params.merge(params_R_df, left_index=True, right_index=True)
    print(f'Treatment params found for spec {i+1}')
    #then get parameters for each outcome 
    for outcome in outcomes: 
        #get outcome data 
        out=spec_df[outcome]
        #tune forest for outcome 
        best_params_Y=tune_rf(parameters, spec_df[spec], out)
        #turn into df 
        params_Y_df=pd.DataFrame.from_dict(best_params_Y, orient='index').rename(columns={0: outcome+'_spec'+str(i+1)})
        #then merge onto final params df
        all_params=all_params.merge(params_Y_df, left_index=True, right_index=True)
        print(f'{outcome} params for spec {i+1} found.')

#write into csv 
all_params.to_csv(data_out/'transformed'/'first_stage_hyperparameters.csv')