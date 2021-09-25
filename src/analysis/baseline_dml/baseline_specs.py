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
#! SET TREATMENT AND OUTCOME
#set for all specifications
#choose outcome 
outcome='chTOTexp'
#choose treatment
treatment='RBTAMT'

#*#########################
#! Specification 1: using MS specification
#*#########################
#specification 1 uses all available observations and hence only variables that have no missings 
#!only for comparison reasons
#* Setup
#choose x and w columns
spec1_xcols=['AGE']
spec1_wcols=['AGE_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']
#! FULL SAMPLE
#clean data and keep only these variables 
subset=variables[variables['comp_samp']==1]
spec1_df=subset[['custid']+[outcome, treatment]+spec1_xcols+spec1_wcols+constants]
#set up DML class
spec1_est=fitDML(spec1_df, treatment, outcome, spec1_xcols)
print('Spec 1 is set up.')

#* Estimation: Linear
#fit linear model 
folds=5
spec1_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec1_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Sparse OLS
#fit linear model 
folds=5
feats=PolynomialFeatures(degree=2, include_bias=False)
spec1_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

#! ONLY HH THAT RECEIVED REBATE
#* Setup
#clean data and keep only these variables 
subset=variables[variables['sample2']==1]
spec1_df=subset[['custid']+[outcome, treatment]+spec1_xcols+spec1_wcols+constants]
#set up DML class
spec1_est=fitDML(spec1_df, treatment, outcome, spec1_xcols)
print('Spec 1 is set up.')
#* Estimation: Linear
#fit linear model 
folds=5
spec1_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec1_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Sparse OLS
#fit linear model 
folds=5
feats=PolynomialFeatures(degree=2, include_bias=False)
spec1_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

#*###### 
#! ANALYSIS
#* PDP Plots
#AGE - linear model
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='lin')
pdp_plot(x_axis_age, y_axis_age, 'AGE')
#AGE - cf model 
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='cf')
pdp_plot(x_axis_age, y_axis_age, 'AGE')

#* ICE Plots
#AGE

#*#########################
#! Specification 2: Add Marriage 
#*#########################
#spec 2 uses only spec 1 variables plus marriage status 

#! FULL SAMPLE
#* Setup
#choose x_cols
spec2_xcols=spec1_xcols+['married']
spec2_wcols=spec1_wcols
#only keep relevant variables 
suebset=variables[variables['compsamp']==1]
spec2_df=subset[['custid']+[treatment, outcome]+spec2_xcols+spec2_wcols+constants]
#set up DML class
spec2_est=fitDML(spec2_df, treatment, outcome, spec2_xcols)
print('Spec 2 is set up.')
#* Estimation: Linear
#fit linear model 
folds=5
spec2_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec2_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Sparse OLS
#fit linear model 
folds=5
#need to define featurizer
feats=PolynomialFeatures(degree=2, include_bias=False)
spec2_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

#*###### 
#! ANALYSIS
#* PDP Plots
#AGE 
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='lin')
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='cf')
pdp_plot(x_axis_age, y_axis_age, 'AGE')

#married

#* ICE Plots
#AGE

#married

#*#########################
#! Specification 3: Liquidity and salary
#*#########################
#add liqudity and salary variables
#! FULL SAMPLE
#* Setup
#choose x_cols 
spec3_xcols=['AGE', 'liqassii', 'married', 'FINCBTXM']
spec3_wcols=spec1_wcols+['FSALARYM']
#only keep relevant variables 
subset=variables[variables['comp_samp']==1]
spec3_df=subset[['custid']+[treatment, outcome]+spec3_xcols+spec3_wcols+constants]
#set up DML class
spec3_est=fitDML(spec3_df, treatment, outcome, spec3_xcols)
print('Spec 3 is set up.')
#* Estimation: Linear
#fit linear model 
folds=5
spec3_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec3_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Sparse OLS
#fit linear model 
folds=5
#need to define featurizer
feats=PolynomialFeatures(degree=2, include_bias=False)
spec3_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

#*###### 
#! ANALYSIS
#* PDP Plots
#AGE 
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='lin')
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='cf')
pdp_plot(x_axis_age, y_axis_age, 'AGE')

#married 

#liquidity 

#salary

#* ICE Plots
#AGE

#married 

#liquidity 

#salary

#*#########################
#! Specification 4: All financial variables
#*#########################
#spec 4 uses all financial variables (smallest sample available)

#! FULL SAMPLE
#* Setup
#choose x_cols 
spec4_xcols=spec3_xcols+['ORGMRTX', 'owned_m', 'notowned', 'QBLNCM1X']
#only keep relevant variables 
subset=variables[variables['comp_samp']==1]
spec4_df=subset[['custid']+[treatment, outcome]+spec4_xcols+constants]
#set up DML class
spec4_est=fitDML(spec4_df, treatment, outcome, spec3_xcols)
print('Spec 4 is set up.')
#* Estimation: Linear
#fit linear model 
folds=5
spec4_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec4_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Sparse OLS
#fit linear model 
folds=5
#need to define featurizer
feats=PolynomialFeatures(degree=2, include_bias=False)
spec4_est.fit_sparseDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

#*###### 
#! ANALYSIS
#* PDP Plots
#AGE 
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='lin')
x_axis_age, y_axis_age=spec1_est.pdp('AGE', model='cf')
pdp_plot(x_axis_age, y_axis_age, 'AGE')

#married 

#liquidity 

#salary

#mortgage

#owned_m

#* ICE Plots
#AGE

#married 

#liquidity 

#salary

#mortgage 

#owned_m
