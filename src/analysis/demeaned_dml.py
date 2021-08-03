from pathlib import Path
import pandas as pd
import numpy as np
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, train_test_split
import utils

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

'''
This file estimates various DML models using demeaned data.
'''

#*#########################
#! FUNCTIONS
#*#########################
def fit_linDML(y_train, t_train, x_train, w_train, params_Y, params_T, n_test=2000): 
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
    linDML=LinearDML(model_y=RFR(max_depth=params_Y['max_depth'], min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), model_t=RFR(max_depth=params_T['max_depth'], min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']))
    #fit to train data 
    linDML.fit(y_train, t_train, X=x_train, W=w_train)
    
    return linDML

def fit_cfDML(y_train, t_train, x_train, w_train, params_Y, params_T, n_test=2000): 
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
    cfDML=CausalForestDML(model_y=RFR(max_depth=params_Y['max_depth'], min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), model_t=RFR(max_depth=params_T['max_depth'], min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']))
    #fit to train data 
    cfDML.fit(y_train, t_train, X=x_train, W=w_train)
    
    return linDML

def coef_var_correlation(X_data, coefs): 
    '''
    Calculate Pearson's coefficients of correlation between observables and individual treatment effects. 
    
    *X_data=X df 
    *coefs=pd series containint treatment effect coefficients
    '''
    #save correlations in dict
    corrs={}
    #for each column in X data calculate correlation with coefficients
    for col in X_data.columns: 
        corrs[col]=np.corrcoef(X_data[col], coefs)[0, 1]
    
    return corrs

#!improve docstring
def marginal_effect_at_means(estimator, x_test, var):
    '''
    Provided a fitted estimator and test data, this function generates a dataset that has all columns at their means except var column and then calculates the constant marginal effect at the means of this var.
    *estimator=fitted DML estimator 
    *x_test=test data 
    *var=variable that MEAM should be calculated for 
    '''
    #get mean X data 
    means=x_test.mean()
    means_df=x_test.copy() 
    for col in x_test.drop(var, axis=1).columns:
        means_df[col]=means[col]
    #then calculate MEAM and get inference df
    meam=estimator.const_marginal_effect_inference(X=means_df)
    #then get summary frame 
    meam_df=meam.summary_frame() 
    
    return meam_df

#*#########################
#! DATA
#*#########################
#read in demeaned data
demeaned_vars=pd.read_csv(data_out/'transformed'/'prepped_data_demeaned.csv')
nan_cols=[]
for col in demeaned_vars.columns: 
    nan_series=demeaned_vars[col].isnull()
    if sum(nan_series)!=0: 
        print(col)
        nan_cols.append(col)
#!some variables have lots of NaN values, need to check this
#!drop for now 
demeaned_vars=demeaned_vars.drop(nan_cols, axis=1)
#choose outcome 
outcome='RBTAMT'
#choose treatment
treatment='RBTAMT'
#split into test and train data and then create subset dataframes
y_train, y_test, r_train, r_test, z_train, z_test=utils.create_test_train(demeaned_vars, outcome, treatment, n_test=2000)
#drop all other expenditure and rebate variables from Z data 
z_train=utils.drop_exp_rbt(z_train)
z_test=utils.drop_exp_rbt(z_test)

#read in hyperparameters for RF - output from tune_first_stage.py
hyperparams=pd.read_csv(data_out/'transformed'/'first_stage_hyperparameters.csv')
#rename hyperparams column 
hyperparams=hyperparams.rename(columns={'Unnamed: 0': 'param'})
#need to turn some entries into integers that are read in str 
hyperparams.loc[0, 'R']=int(hyperparams.loc[0, 'R'])
hyperparams.loc[2, 'R']=int(hyperparams.loc[2, 'R'])
hyperparams.loc[0, 'Y']=int(hyperparams.loc[0, 'Y'])
hyperparams.loc[2, 'Y']=int(hyperparams.loc[2, 'Y'])

#*Setup
#for now consider a subset of observables as X; for definitions see MS_columnexplainers.numbers
x_cols=['AGE', 'FAM_SIZE', 'children', 'FINCBTXM', 'FSALARYM']+['MARITAL1_'+str(num) for num in range(1, 5)]+['SAVA_CTX_'+lit for lit in ['A', 'B', 'C', 'D']]
#split Z into X and W data
x_train, w_train=utils.split_XW(Z=z_train, x_columns=x_cols)
x_test, z_test=utils.split_XW(Z=z_test, x_columns=x_cols)

#*Random Forest hyperparameters
#turn parameter df into respective parameter dictionaries 
best_params_R={param: val for param, val in zip(hyperparams['param'], hyperparams['R'])}
best_params_Y={param: val for param, val in zip(hyperparams['param'], hyperparams['Y'])}

#*####
#! Linear DML
'''
Here consider DML model with semi-parametric specification: 
Y_{it}=\theta*R_{it} + g(X_{it}, W_{it}) + U_{it}
R_{it}=f(X_{it}, W_{it}) + V_{it}
For estimation of g() and f() a regression forest is used (for now - explore other options later on) 
The hyperparameters for the forest are tuned in tune_first_stage.py and contained in 'hyperparams_rf' dataframe
'''
#fit linear DML model to train data
linDML=fit_linDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R)
#get constant marginal effect 
cme_inf_lin=linDML.const_marginal_effect_inference(X=x_test)
cme_df_lin=cme_inf_lin.summary_frame()
ate_inf_lin=linDML.ate_inference(X=x_test)
#everythign significant and ATE also significant 

#*Marginal Effect at the Means
meam_age=marginal_effect_at_means(linDML, x_test, 'AGE')
meam_children=marginal_effect_at_means(linDML, x_test, 'children')

#*####
#! Causal Forest DML
'''
Here consider DML model with a completely nonparametric specification. To do so use a Causal Forest (Athey and Wager (2016)) that fits following local moment equation: 
E[(Y_{it} - E[Y_{it}|X_{it}, W_{it}] - \theta(X)*(T - E[T|X, W]) - beta(x)) (T_{it};1) | X_{it}=x] = 0  

For estimation of the first stage use a random forest again with same hyperparameters as with linear DML.
'''
#fit causal forest DML to train data 
cfDML=fit_cfDML(y_train, r_train, x_train, w_train, best_params_Y, best_params_R)
#get constant marginal effect 
cme_inf_cf=cfDML.const_marginal_effect_inference(X=x_test)
cme_df_cf=cme_inf_cf.summary_frame()

#*Marginal Effect at the Means
meam_age=marginal_effect_at_means(cfDML, x_test, 'AGE')
meam_children=marginal_effect_at_means(cfDML, x_test, 'children')
