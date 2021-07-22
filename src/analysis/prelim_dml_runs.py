from pathlib import Path
import numpy as np
from numpy.random import default_rng
import pandas as pd 
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

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
    rng=default_rng()
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

#TODO: streamline this process and split into smaller steps 
def LinDML_rf(Y, T, W, X, params_Y, params_T, n_test=2000): 
    '''
    Estimate a partially linear model using the DML approach. Estimation of E[Y|X] and E[T|X] is done using a random forest. 

    *Y=outcome data 
    *T=treatment data 
    *W=confounders that have no effect on treatment effect 
    *X=confounders that can have effect on treatment effect 
    *params=parameters for random forest
    *n_test=size of test set
    '''
    #first split data into training and test set 
    Y_test, Y_train, R_test, R_train, X_test, X_train, W_test, W_train=split_data(Y, T, X, W, n_test=n_test)
    #initialize DML model 
    linDML=LinearDML(model_y=RFR(max_depth=params_Y['max_depth'], min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), model_t=RFR(max_depth=params_T['max_depth'], min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']))
    #fit to train data 
    linDML.fit(Y_train, R_train, X=X_train, W=W_train)
    
    #also return X_test set for futher use
    return linDML, X_test

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
#! ATE ANALYSIS
#*#########################
#*examples on application of econml.dml: https://github.com/microsoft/EconML
'''
This section contains several approaches to estimate the Average Treatment Effect (ATE) of the 2008 tax rebate on changes in consumption. 
For now look at change in total expenditures as outcome and total rebate amount as treatment. 
Define: 
R=treatment -> rebate variable of choice
Y=outcome -> as mentioned: change in total expenditure
X=observables that have effect on Y and R & are leading to heterogeneity in treatment effect 
W=confounders that have effect on Y and R

The parametric model in Parker et al. (2013) (also homogenous model in Misra & Surico (2014)) looks like this: 
Y_{it}=Month Dummies + beta*R_{it} + gamma*X_{it} + U_{it}
-> try to replicate their results later on 

Each subsection contains sketch of model estimated. For more consult docs on estimation and methods (coming soon...).
'''
Y=expenditures['chTOTexp']
R=treatment['RBTAMT']
#indicator variable if rebate received as treatment
iR=treatment['iREB']

#*####
#! Linear DML
'''
Here consider DML model with semi-parametric specification: 
Y_{it}=\theta*R_{it} + g(X_{it}, W_{it}) + U_{it}
R_{it}=f(X_{it}, W_{it}) + V_{it}
For estimation of g() and f() a regression forest is used (for now - explore other options later on) -> probably need to tune the forest before such that it is optimal model and then supply to LinearDML()

Open Questions: 
- how does DML estimator distinct i & t dimensions? 
    - is this even relevant? 
- what variables should be in X, what variables only in W?
- what model should be used for f() and g()?
'''

#*Setup
#for now consider a subset of observables as X; for definitions see MS_columnexplainers.numbers
X=observables[['AGE', 'FAM_SIZE', 'ORGMRTX', 'children', 'FINCBTXM', 'FSALARYM']+['MARITAL1_'+str(num) for num in range(1, 6)]+['SAVA_CTX_'+lit for lit in ['A', 'B', 'C', 'D', 'T']]]
#use rest of observables as W 
W=observables[[col for col in observables.columns if col not in X.columns]]

#*Random Forest Tuning 
#find optimal parameters for random forests used for Y and R 
#set parameters to look at 
parameters={'max_depth': [None, 5, 10, 15, 20], 'min_samples_leaf': [10, 20, 60], 'max_features': ['auto', 'sqrt', 'log2']}
#rf for Y
best_params_Y=tune_rf(parameters, X=X, Y=Y)
print('Got RF params for Y')
#rf for R 
best_params_R=tune_rf(parameters, X=X, Y=R)
print('Got RF params for R')

#*Estimation
#estimate linear DML model and get ATE and constant marginal effect (CME) inference 
ate_inference, cme_inference, X_test=LinDML_rf(Y, R, W, X, best_params_Y, best_params_R)
#inverstigate the cme inference results 
#CME summary dataframe that contains inference for each individual 
cme_df=cme_inference.summary_frame()
#then check correlation between point_estimates and X 
correlations=coef_var_correlation(X_test, cme_df['point_estimate'])
#calculate correlations only for entries that have significant point_estimates, i.e. for who are the point estimates significant?
#filter X_test for rows that have a significant point estaimte 
#look at 10% level -> 308 observations
cme_df_s=cme_df[cme_df['pvalue']<=0.1]
X_significant=X_test.iloc[cme_df_s.index, :]
significance_corrs=coef_var_correlation(X_significant, cme_df_s['point_estimate'])

#*Visualization 
fig, ax=plt.subplots()
#show correlation between X variables and theta(X)
ax.scatter(X_test['MARITAL1_1'], cme_df['point_estimate'], s=0.5)

#*Effect of only one variable 
#to visualize effect as function of e.g. income take average of all other variables as value for all observations and keep only income as varying across households, then estimate and plot 
#does this make sense? 
