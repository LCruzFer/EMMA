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
fig_out=wd.parents[2]/'figures'
'''
This file estimates the baseline specifications of the linear DML model to estimate the constant marginal Conditional Average Treatment Effect (CATE), which is the MPC as a function of households characteristics.
'''

#*#########################
#! IDEAS & TO DOs
#*#########################
#IDEA: control for how many months after announcement rebate was received 
#*#########################
#! FUNCTIONS
#*#########################
def split_sample(df):
    ind=np.random.uniform(low=0, high=len(df), size=int(len(df)/2)).astype(int)
    s1=df.iloc[ind, :]
    s2=df.iloc[~ind, :]
    return s1, s2

def z_test(b, b0, se1, se2): 
    '''
    Calculate SE test.
    '''
    test=(b-b0)/np.sqrt((se1**2+se2**2))
    return test

def pdp_plot(x_axis, y_axis, var, model): 
    '''
    Create partial dependence plot using x- and y-axis values taken from estimation. 
    *x_axis=1-dimensional numpy array, x-axis values
    *y_axis=2-dimensional numpy array, y-axis values (point estimate and CI bounds)
    *var=str; variable name for title and axis
    *model=str; model name, use 'linear', 'causal forest' or 'sparse linear' 
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
    ax.set_title(f'Partial Dependence Plot for {var} using {model} model')
    figname='PDP_'+var+'_model'
    plt.savefig(fig_out/'PDP'/figname)
    return fig, ax

def ice_plot(x_axis, y_axis, var, model):
    '''
    Create Individal Conditional Expecation plot using x- and y-axis values taken from estimation.
    *x_axis=1-dimensional numpy array, x-axis values
    *y_axis=2-dimensional numpy array, y-axis values (point estimate and CI bounds)
    *var=str; variable name for title and axis
    *model=str; model name, use 'linear', 'causal forest' or 'sparse linear' 
    '''
    #! HERE LOTS OF ROOM FOR IMPROVEMENT
    fig, ax=plt.subplots()
    for i in range(y_axis.shape[1]):
        ax.plot(x_axis, y_axis[:, i, 1])
    ax.set_xlabel(var)
    ax.set_ylabel('ICE of MPC')
    ax.set_title(f'Individual Conditional Expectation Plot for {var} using {model} model')
    figname='ICE_'+var+'_model'
    plt.savefig(fig_out/'ICE'/'figname')
    return fig, ax

def get_cdf(data): 
    '''
    Supply numpy array, pandas series or list of data and return proportional value and sorted values, i.e. CDF with x and y values.
    '''
    #point estimates sorted by size 
    pe_sorted=np.sort(np.array(data))
    #proportional values of sample 
    proportions=np.arange(len(pe_sorted))/(len(pe_sorted)-1)
    
    return(pe_sorted, proportions)

def cdf_figure(cdf, l_cdf, u_cdf, spec): 
    '''
    Plot empirical CDF with confidence interval bounds.
    *cdf=cdf of point estimate 
    *l_cdf=cdf of lower bound of ci 
    *u_cdf=cdf of upper bound of ci 
    '''
    fig, ax=plt.subplots()
    ax.plot(cdf[0], cdf[1])
    ax.plot(l_cdf[0], l_cdf[1])
    ax.plot(u_cdf[0], u_cdf[1])
    figname=spec+'_cdf'
    plt.savefig(fig_out/'CDF'/figname)
    return(fig, ax)

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
outcome='chNDexp'
#choose treatment
treatment='RBTAMT'

#*#########################
#! Specification 1: using MS specification
#*#########################
#specification 1 uses all available observations and hence only variables that have no missings 
#!only for comparison reasons
#* Setup
#choose x and w columns
spec1_xcols=['AGE', 'AGE_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']
#spec1_wcols=['AGE_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']
#! FULL SAMPLE
#clean data and keep only these variables 
subset=variables[variables['comp_samp']==1]
#! w cols missing right now because None
spec1_df=subset[['custid']+[outcome, treatment]+spec1_xcols+constants]
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

#* Test ITE-ATE 
#! need to split test sample into two: one for ATE, one for ITE, ow they are not independent 
xtest1, xtest2=split_sample(spec1_est.x_test)
#get ate inference in first sample
ate_inf=spec1_est.cfDML.const_marginal_ate_inference(X=xtest1)
#fit model to second sample
cate_ate_df=spec1_est.cfDML.const_marginal_effect_inference(X=xtest2).summary_frame()
cate_ate_df['ATE']=ate_inf.mean_point
#adjust SE of ATE for sample size
cate_ate_df['ATE_stderr']=ate_inf.stderr_mean/len(xtest1)
#construct a hypothetical t-test 
cate_ate_df['zstat_ateite']=z_test(cate_ate_df['point_estimate'], cate_ate_df['ATE'], cate_ate_df['stderr'], cate_ate_df['ATE_stderr'])
sum(cate_ate_df['zstat_ateite'].abs()>1.64)

#* PDP Plots
#linear model 
for var in spec1_xcols: 
    x_axis, y_axis=spec1_est.pdp(var, model='lin')
    pdp_plot(x_axis, y_axis, var, 'linear')

#cf model 
for var in spec1_xcols:
    x_axis, y_axis=spec1_est.pdp(var, model='cf')
    pdp_plot(x_axis, y_axis, var, 'causal forest')

#* ICE Plots
for var in spec1_xcols: 
    y_axis.shape
    var='AGE'
    x_axis, y_axis=spec1_est.ice(var, model='lin')
    ice_plot(x_axis, y_axis, var, 'linear')
#cf model 
for var in spec1_xcols:
    var='AGE'
    x_axis, y_axis=spec1_est.ice(var, model='cf')
    ice_plot(x_axis, y_axis, var, 'causal forest')

#* CDF Plots 
cdf_pe=get_cdf(spec1_est.lin_cate_df['point_estimate'])
cdf_lci=get_cdf(spec1_est.lin_cate_df['ci_lower'])
cdf_uci=get_cdf(spec1_est.lin_cate_df['ci_upper'])
cdf_figure(cdf_pe, cdf_lci, cdf_uci)

#*#########################
#! Specification 2: Add Marriage 
#*#########################
#spec 2 uses only spec 1 variables plus marriage status 

#! FULL SAMPLE
#* Setup
#choose x_cols
spec2_xcols=spec1_xcols+['married']
#spec2_wcols=spec1_wcols
#only keep relevant variables 
suebset=variables[variables['comp_samp']==1]
#! no W cols right now !
spec2_df=subset[['custid']+[treatment, outcome]+spec2_xcols+constants]
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
#linear
for var in spec2_xcols:
    x_axis, y_axis=spec2_est.pdp(var, model='lin')
    pdp_plot(x_axis, y_axis, var, 'linear')
#causal forest 
for var in spec2_xcols:
    x_axis, y_axis=spec2_est.pdp(var, model='cf')
    pdp_plot(x_axis, y_axis, var, 'causal forest')

#* ICE Plots
for var in spec2_xcols:
    x_axis, y_axis=spec2_est.ice(var, model='cf')
    ice_plot(x_axis, y_axis, var, 'causal forest')

#* CDF Plots 
cdf_pe=get_cdf(spec2_est.cf_cate_df['point_estimate'])
cdf_lci=get_cdf(spec2_est.cf_cate_df['ci_lower'])
cdf_uci=get_cdf(spec2_est.cf_cate_df['ci_upper'])
cdf_figure(cdf_pe, cdf_lci, cdf_uci)

#*#########################
#! Specification 3: Liquidity and salary
#*#########################
#add liqudity and salary variables
#! FULL SAMPLE
#* Setup
#choose x_cols 
spec3_xcols=['AGE', 'liqassii', 'married', 'FINCBTXM', 'FSALARYM']
#only keep relevant variables 
subset=variables[variables['l_samp']==1]
#!no w cols right now
spec3_df=subset[['custid']+[treatment, outcome]+spec3_xcols+constants]
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
#linear
for var in spec3_xcols:
    x_axis, y_axis=spec1_est.pdp(var, model='lin')
    pdp_plot(x_axis, y_axis, var, 'linear')
#causal forest 
for var in spec3_xcols:
    x_axis, y_axis=spec1_est.pdp(var, model='cf')
    pdp_plot(x_axis, y_axis, var, 'causal forest')

#* ICE Plots
for var in spec2_xcols:
    var='AGE'
    x_axis, y_axis=spec3_est.ice(var, model='lin')
    ice_plot(x_axis, y_axis, var, 'linear')

for var in spec2_xcols:
    var='AGE'
    x_axis, y_axis=spec3_est.ice(var, model='cf')
    ice_plot(x_axis, y_axis, var, 'causal forest')

#* CDF Plots 
cdf_pe=get_cdf(spec3_est.cf_cate_df['point_estimate'])
cdf_lci=get_cdf(spec3_est.cf_cate_df['ci_lower'])
cdf_uci=get_cdf(spec3_est.cf_cate_df['ci_upper'])
cdf_figure(cdf_pe, cdf_lci, cdf_uci)

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
for var in spec3_xcols:
    x_axis, y_axis=spec1_est.pdp(var, model='lin')
    pdp_plot(x_axis, y_axis, var, 'linear')
#causal forest 
for var in spec3_xcols:
    x_axis, y_axis=spec1_est.pdp(var, model='cf')
    pdp_plot(x_axis, y_axis, var, 'causal forest')

#* ICE Plots
