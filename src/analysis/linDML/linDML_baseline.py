import sys
from pathlib import Path
import pandas as pd
import numpy as np
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, train_test_split
import itertools
#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils as du 
from utils import estim_utils as eu

#* set data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'

'''
This file estimates various DML models.
'''

#*#########################
#! FUNCTIONS
#*#########################
def fit_linDML(y_train, t_train, x_train, w_train, params_Y, params_T, folds, n_test=2000): 
    '''
    Estimate a partially linear model using the DML approach. Estimation of E[Y|X] and E[T|X] is done using a random forest. 

    *Y=outcome data 
    *T=treatment data 
    *W=confounders that have no effect on treatment effect 
    *X=confounders that can have effect on treatment effect 
    *params=parameters for random forest
    *n_test=size of test set
    '''
    #initialize DML model with tuned parameters
    linDML=LinearDML(model_y=RFR(max_depth=params_Y['max_depth'], min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), model_t=RFR(max_depth=params_T['max_depth'], min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']), cv=folds)
    print('Model set up!')
    #fit to train data 
    linDML.fit(y_train, t_train, X=x_train, W=w_train)
    print('Model fitted!')
    return linDML

#*#########################
#! DATA
#*#########################
#read in data
variables=pd.read_csv(data_out/'transformed'/'cleaned_dummies.csv')
variables=variables.replace(np.nan, 0)

#* Random Forest Hyperparameters
#read in hyperparameters for RF - output from tune_first_stage.py
hyperparams=pd.read_csv(data_out/'transformed'/'first_stage_hyperparameters.csv')
#rename hyperparams column 
hyperparams=hyperparams.rename(columns={'Unnamed: 0': 'param'})
#need to turn some entries into integers that are read in str 
hyperparams.loc[0, 'R']=int(hyperparams.loc[0, 'R'])
hyperparams.loc[2, 'R']=int(hyperparams.loc[2, 'R'])
hyperparams.loc[0, 'Y']=int(hyperparams.loc[0, 'Y'])
hyperparams.loc[2, 'Y']=int(hyperparams.loc[2, 'Y'])
#turn parameter df into respective parameter dictionaries 
best_params_R={param: val for param, val in zip(hyperparams['param'], hyperparams['R'])}
best_params_Y={param: val for param, val in zip(hyperparams['param'], hyperparams['Y'])}

#*Setup
#choose outcome 
outcome='chTOTexp'
#choose treatment
treatment='RBTAMT'
#set index to newid 
#split into test and train data and then create subset dataframes
y_train, y_test, r_train, r_test, z_train, z_test=du().create_test_train(variables, 'custid', outcome, treatment, n_test=500)
#for now consider a subset of observables as X; for definitions see MS_columnexplainers.numbers
x_cols=['AGE', 'children', 'QESCROWX', 'fam_salary_inc', 'tot_fam_inc',
        'adults', 'top99pc', 'bothtop99pc', 'CUTENURE', 'validIncome', 
        'validAssets', 'payment']+['maritalstat_'+str(num) for num in range(1, 5)]+['totbalance_ca_'+ let for let in ['A', 'B', 'C', 'D']]
#split Z into X and W data
x_train, w_train=du().split_XW(Z=z_train, x_columns=x_cols)
x_test, w_test=du().split_XW(Z=z_test, x_columns=x_cols)
#drop id variables from w data 
w_train=w_train.drop(['custid', 'year'], axis=1)
w_test=w_test.drop(['custid', 'year'], axis=1)
print('Data is ready!')

#*#########################
#! PANEL LINEAR DML
#*#########################
'''
Here consider DML model with semi-parametric specification: 
Y_{it}=\theta*R_{it} + g(X_{it}, W_{it}) + U_{it}
R_{it}=f(X_{it}, W_{it}) + V_{it}
For estimation of f() and conditional mean of Y a random forest is used.
Panel DML estimator by Chernozhukov et al (2021) is same procedure but based on different folds in cross-fitting.
'''
#get folds for first-stage cross fitting
x_train=x_train.reset_index(drop=True) 
x_test=x_test.reset_index(drop=True)
w_train=w_train.reset_index(drop=True)
w_test=w_test.reset_index(drop=True) 
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
fs_folds=eu().cross_fitting_folds(w_train, 5, 6, 'quarter')
#fit linear DML model to train data
linDML=fit_linDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R, folds=fs_folds)
print('Linear DML fitted')
#get constant marginal effect 
cme_inf_lin=linDML.const_marginal_effect_inference(X=x_test)
#get summary dataframe
cme_df_lin=cme_inf_lin.summary_frame()
print(len(cme_df_lin[cme_df_lin['pvalue']<=0.1]))

#write to csv
filename='CME_'+treatment+'.csv'
cme_df_lin.to_csv(data_out/'results'/filename)

#*Marginal Effect at the Means
#apply get_all_meam() to x_test data to get MEAM for all variables 
meams=eu().get_all_meam(linDML, x_test)
for col in x_train.columns: 
    sig=meams[meams['pvalue'+'_'+col]<=0.1]
    print(f'{col}: {len(sig)}')

#write to csv 
filename_meam='MEAMS_'+treatment+'.csv'
meams.to_csv(data_out/'results'/filename_meam, index=False)