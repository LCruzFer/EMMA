import time
import sys
from pathlib import Path
from typing import AsyncGenerator
from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils
from utils import fitDML

#* set data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'

'''
This file estimates the baseline specifications of the linear DML model to estimate the constant marginal Conditional Average Treatment Effect (CATE), which is the MPC as a function of households characteristics.
'''

#*#########################
#! IDEAS & TO DOs
#*#########################

#IDEA: use percentile of rebate size household is in to control for what role the size of the income shock has on MPC
#TODO: how to control for different channels 
#TODO: what is X, what is W? 
#TODO: streamline for final estimation
#IDEA: control for how many months after announcement rebate was received 

#*#########################
#! FUNCTIONS
#*#########################
def pdp_plot(x_axis, y_axis, var): 
    '''
    Create partial dependence plot using x- and y-axis values taken from estimation. 
    *x_axis=1-dimensional numpy array, x-axis values
    *y_axis=2-dimensional numpy array, y-axis values (point estimate and CI bounds)
    *var=str; variable name for title and axis
    '''
    #set up figure
    fig, ax=plt.subplots()
    #set up colors
    colors=['red', 'blue', 'red']
    #set up labels
    labels=['Lower CI', 'Point Estimate', 'Upper CI']
    for i, col, lab in zip(range(y_axis.shape[1]-1), colors, labels):
        ax.plot(x_axis, y_axis[:, i], color=col, label=lab)
    #add legend
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')
    #add shaded area between CI lines
    ax.fill_between(x_axis, y_axis[:, 0], y_axis[:, 2], color='orangered', alpha=0.2, where=y_axis[:, 4]>=0.1)
    ax.fill_between(x_axis, y_axis[:, 0], y_axis[:, 2], color='mediumseagreen', alpha=0.2, where=y_axis[:, 4]<=0.1)
    #set axis labels and title 
    ax.set_xlabel(var)
    ax.set_ylabel('MPC')
    ax.set_title(f'Partial Dependence Plot for {var}')
    
    return fig, ax

def coef_var_correlation(df, coefs): 
    '''
    Calculate Pearson's coefficients of correlation between observables and individual treatment effects. 
    
    *df=pd dataframe; X test data 
    *coefs=pd series; treatment effect coefficients
    '''
    #save correlations in dict
    corrs={}
    #for each column in X data calculate correlation with coefficients
    for col in df.columns: 
        corrs[col]=np.corrcoef(df[col], coefs)[0, 1]
    
    return corrs

#*#########################
#! DATA
#*#########################
#read in data
variables=pd.read_csv(data_out/'transformed'/'prepped_data.csv')
print('Variables loaded')

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

#* Month constants used in every spec 
#relative to first month
constants=['const'+str(i) for i in range(1, 15)]

#*#########################
#! Specification 1: using MS specification
#*#########################

#*###### 
#! SETUP
#specification 1 uses all available observations and hence only variables that have no missings 
#* Choosing treatment, outcome and x_cols
#choose outcome 
outcome='chTOTexp'
#choose treatment
treatment='RBTAMT'
#choose x_cols
ms_xcols=['AGE', 'AGE_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']
#clean data and keep only these variables 
spec1_df=variables[['custid']+[outcome, treatment]+ms_xcols+constants]
#set up DML class
spec1_est=fitDML(spec1_df, treatment, outcome, ms_xcols)
print('Spec 1 is set up.')

#*###### 
#! ESTIMATION

#* Estimation: Linear
#fit linear model 
folds=5
spec1_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
spec1_est.lin_cate_df

#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec1_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
spec1_est.cfDML.const_marginal_ate_interval(X=spec1_est.x_test, alpha=0.1)

x_axis_cf, y_axis_cf=spec1_est.pdp('AGE', model='cf', alpha=0.1)

pdp_plot(x_axis_cf, y_axis_cf, 'AGE')

#* Estimation: Sparse OLS
#fit linear model 
folds=5
feats=PolynomialFeatures(degree=2, include_bias=False)
spec1_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
spec1_est.sp_cate_df

#*###### 
#! ANALYSIS
#* PDP and ICE plots
#calculate PDP values
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='lin')
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='cf')
pdp_plot(x_axis_age, y_axis_age, 'AGE')

#*#########################
#! Specification 2: using more variables 
#*#########################
#specification 1 uses all available observations and hence only variables that have no missings 
#* Choosing treatment, outcome and x_cols
#choose outcome 
outcome='chTOTexp'
#choose treatment
treatment='RBTAMT'
#choose x_cols
spec2_xcols=['AGE', 'AGE_sq','chFAM_SIZE', 'chFAM_SIZE_sq', 'liqassii', 'married', 'FSALARYM']
#only keep relevant variables 
spec2_df=variables[['custid']+[treatment, outcome]+spec2_xcols+constants]
spec2_df=spec2_df.dropna()
#set up DML class
spec2_est=fitDML(spec2_df, treatment, outcome, spec2_xcols)
print('Spec 2 all set up.')

#* Estimation: Linear
#fit linear model 
folds=5
spec2_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
spec2_est.linDML.const_marginal_ate_inference(X=spec2_est.x_test).stderr_mean

spec2_est.lin_cate_df['ci_lower'].mean()
spec2_est.pdp_ice('AGE', 'lin')
pdp_plot()
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec2_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
test=spec2_est.cf_cate_df
test['significant']=(test['pvalue']<0.1).astype(int)

#* Estimation: Sparse OLS
#fit linear model 
folds=5
feats=PolynomialFeatures(degree=2, include_bias=False)
spec2_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)