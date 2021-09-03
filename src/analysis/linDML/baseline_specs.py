import time
import sys
from pathlib import Path
from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
from econml.dml import LinearDML, CausalForestDML
from econml.inference import BootstrapInference
from sklearn.ensemble import RandomForestRegressor as RFR
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
This file estimates the 3 baseline specifications of the linear DML model to estimate the constant marginal Conditional Average Treatment Effect (CATE), which is the MPC as a function of households characteristics.
'''

#*#########################
#! FUNCTIONS
#*#########################
def fit_linDML(y_train, t_train, x_train, w_train, params_Y, params_T, folds): 
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
    linDML=LinearDML(model_y=RFR(n_estimators=1000,
                                max_depth=params_Y['max_depth'],       
                                min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), 
                    model_t=RFR(n_estimators=1000,
                                max_depth=params_T['max_depth'],    
                                min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']), 
                    cv=folds, fit_cate_intercept=False, 
                    random_state=2021
                    )
    print('Model set up!')
    #fit to train data 
    linDML.fit(y_train, t_train, X=x_train, W=w_train, 
            #bootstrapped or asymptotic inference?
                #inference=BootstrapInference(n_bootstrap_samples=50)
                )
    print('Model fitted!')
    return linDML

def fit_cfDML(y_train, t_train, x_train, w_train, params_Y, params_T, folds): 
    '''
    Estimate a nonparametric model using the DML approach. Estimation of E[Y|X] and E[T|X] is done using a random forest. 

    *Y=outcome data 
    *T=treatment data 
    *W=confounders that have no effect on treatment effect 
    *X=confounders that can have effect on treatment effect 
    *params=parameters for random forest
    *n_test=size of test set
    '''
    #initialize DML model with tuned parameters
    cfDML=CausalForestDML(model_y=RFR(n_estimators=300,
                                max_depth=params_Y['max_depth'],       
                                min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), 
                    model_t=RFR(n_estimators=300,
                                max_depth=params_T['max_depth'],    
                                min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']), 
                    #cv=folds,
                    random_state=2021, 
                    n_estimators=5000, 
                    drate=False, 
                    )
    print('Model set up!')
    #fit to train data 
    cfDML.fit(y_train, t_train, X=x_train, W=w_train, 
            #bootstrapped or asymptotic inference?
                #inference=BootstrapInference(n_bootstrap_samples=50)
                )
    print('Model fitted!')
    return cfDML

#*#########################
#! DATA
#*#########################
#read in data
#variables=pd.read_csv(data_out/'transformed'/'cleaned_dummies.csv')
#variables=variables.replace(np.nan, 0)
#panel structure data 
variables=pd.read_csv(data_out/'transformed'/'panel_w_lags.csv')
print('Variables loaded')

#* Choosing treatment and outcome
#choose outcome 
outcome='chTOTexp'
#choose treatment
treatment='RBTAMT'

#* Splitting Data into test and train sample
#split into test and train data and then create subset dataframes
y_train, y_test, r_train, r_test, z_train, z_test=du().create_test_train(variables, 'custid', outcome, treatment, n_test=500)
#drop change variables from confounders
z_train=z_train.drop([col for col in z_train.columns if 'ch' in col], axis=1)
z_test=z_test.drop([col for col in z_test.columns if 'ch' in col], axis=1)
#merge children back onto Z datasets
z_train=z_train.merge(variables[['newid', 'children']], on='newid')
z_test=z_test.merge(variables[['newid', 'children']], on='newid')

#merge lags of treatment and rebate back on Z data for spec 3, others do not include them
z_train_spec3=z_train.merge(variables[['newid', outcome+'_lag', treatment+'_lag']], on='newid')
z_test_spec3=z_test.merge(variables[['newid', outcome+'_lag', treatment+'_lag']], on='newid')

#* Splitting Z into X and W (Spec 1 and 2)
#consider a subset of observables as X; for definitions see MS_columnexplainers.numbers
x_cols=['AGE', 'children', 'fam_salary_inc', 'tot_fam_inc',
        'adults', 'payment', 'married', 'owned_nm', 'owned_m']
#split Z into X and W data
x_train, w_train=du().split_XW(Z=z_train, x_columns=x_cols)
x_test, w_test=du().split_XW(Z=z_test, x_columns=x_cols)
#drop id variables from w data 
w_train=w_train.drop(['custid', 'year', 'interviewno', 'newid', 'month'], axis=1)
w_test=w_test.drop(['custid', 'year', 'interviewno', 'newid', 'month'], axis=1)

#* Splitting Z into X and W (Spec 3)
#split Z into X and W data
x_train_spec3, w_train_spec3=du().split_XW(Z=z_train, x_columns=x_cols)
x_test_spec3, w_test_spec3=du().split_XW(Z=z_test, x_columns=x_cols)
#drop id variables from w data 
w_train_spec3=w_train_spec3.drop(['custid', 'year', 'interviewno', 'newid', 'month'], axis=1)
w_test_spec3=w_test_spec3.drop(['custid', 'year', 'interviewno', 'newid', 'month'], axis=1)
print('Data is ready!')
#* Saving x_test 
#for partial dependence plots need the x test set, save it as csv 
x_test.to_csv(data_out/'transformed'/'x_testset.csv', index=False)

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
print('Hyperparameters ready')

#*#########################
#! ESTIMATION: Specification 1
#*#########################
'''
Here consider DML model with semi-parametric specification: 
Y_{it}=\theta*R_{it} + g(X_{it}, W_{it}) + U_{it}
R_{it}=f(X_{it}, W_{it}) + V_{it}
For estimation of f() and conditional mean of Y a random forest is used.
Original DML estimator assuming strict exogeneity is used here. Not accounting for any lag/panel structure.
'''
#set how many folds are used in cross-fitting
n_folds=2
#fit linear DML model to train data
tik=time.time()
spec1=fit_linDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R, folds=n_folds)
tok=time.time() 
print(tok-tik)
print('Linear DML fitted')
#get constant marginal CATE summary df
spec1_inf=spec1.const_marginal_effect_inference(X=x_test).summary_frame()
print(len(spec1_inf[spec1_inf['pvalue']<=0.1]))
#get ATE 
spec1_ate=spec1.ate_inference(X=x_test)
#insignificant! (nice)
#! meam testing area 
meams=eu().get_all_meam(spec1, x_test)
for var in x_cols: 
    print(var, len(meams[meams['pvalue_'+var]<=0.1]))
#write to csv 
spec1_inf.to_csv(data_out/'results'/'cate_spec1.csv', index=False)

#*#########################
#! ESTIMATION: Specification 2
#*#########################
'''
Panel DML estimator by Chernozhukov et al (2021) is used here but without lags, i.e. using algorithm but still assuming strict exogeneity - which should lead to everything working out because less strict assumption.
'''
#get folds for first-stage cross fitting
#need to reset indices, ow LinearDML's fold arguments makes problems
x_train=x_train.reset_index(drop=True) 
x_test=x_test.reset_index(drop=True)
w_train=w_train.reset_index(drop=True)
w_test=w_test.reset_index(drop=True) 
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
#get folds
fs_folds=eu().cross_fitting_folds(w_train, 5, 6, 'quarter')

#fit linear DML model to train data
tik=time.time()
spec2=fit_linDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R, folds=fs_folds)
tok=time.time() 
print(tok-tik)
print('Linear DML fitted')
#get constant marginal CATE summary df
spec2_inf=spec2.const_marginal_effect_inference(X=x_test).summary_frame()
#write to csv 
spec2_inf.to_csv(data_out/'results'/'cate_spec2.csv')

#*#########################
#! ESTIMATION: Specification 3
#*#########################
'''
Panel DML estimator by Chernozhukov et al (2021) with lag structure.
'''
#get folds for first-stage cross fitting
#need to reset indices, ow LinearDML's fold arguments makes problems
x_train=x_train.reset_index(drop=True) 
x_test=x_test.reset_index(drop=True)
w_train=w_train.reset_index(drop=True)
w_test=w_test.reset_index(drop=True) 
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
#get folds
fs_folds=eu().cross_fitting_folds(w_train, 5, 6, 'quarter')

#fit linear DML model to train data
tik=time.time()
spec3=fit_linDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R, folds=fs_folds)
tok=time.time() 
print(tok-tik)
print('Linear DML fitted')
#get constant marginal CATE
spec3_inf=spec3.const_marginal_effect_inference(X=x_test).summary_frame()
#write to csv 
spec3_inf.to_csv(data_out/'results'/'cate_spec3.csv')

#*#########################
#! ESTIMATION: Specification 4
#*#########################
#fit Causal Forest DML model to train data
tik=time.time()
spec4=fit_cfDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R, folds=5)
tok=time.time() 
print(tok-tik)
print('CF DML fitted')
#get constant marginal CATE
spec4_inf=spec4.const_marginal_effect_inference(X=x_test).summary_frame()
print(len(spec4_inf[spec4_inf['pvalue']<=0.1]))
#write to csv 
spec4_inf.to_csv(data_out/'results'/'cate_spec4.csv')