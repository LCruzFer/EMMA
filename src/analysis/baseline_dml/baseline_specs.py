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
def retrieve_params(params_df, treatment, outcome, spec): 
    '''
    Get first stage parameters as dictionary for specification and setting.
    *params_df=pandas df, output of tune_first_stage, index must be parameter name!
    *treatment=str, treatment name 
    *treatment=str, outcome name 
    *spec=str, must be specX where X is number of specification
    '''
    treat_col=treatment+'_'+spec
    spec_col=outcome+'_'+spec
    params_T={param: val for param, val in zip(params_df.index, params_df[treat_col])}
    params_Y={param: val for param, val in zip(params_df.index, params_df[spec_col])}
    return params_T, params_Y

def split_sample(df):
    ind=np.random.uniform(low=0, high=len(df), size=int(len(df)/2)).astype(int)
    s1=df.iloc[ind, :]
    s2=df.iloc[~ind, :]
    return s1, s2

def z_test(b, b0, se1, se2): 
    '''
    Calculate z teststatistic.
    '''
    test=(b-b0)/np.sqrt((se1**2+se2**2))
    return test

def ite_ate_test(spec, model): 
    #! THIS NEEDS IMPROVEMENT
    '''
    z-test whether ITE is significantly different from ATE.
    *spec=fitDML object
    *model=str, choose which model should be looked at
    '''
    #get estimator depending on model
    estim=spec.selectmodel(model)
    xtest1, xtest2=split_sample(spec.x_test)
    #get ate inference in first sample
    ate_inf=estim.const_marginal_ate_inference(X=xtest1)
    #fit model to second sample
    cate_ate_df=estim.const_marginal_effect_inference(X=xtest2).summary_frame()
    cate_ate_df['ATE']=ate_inf.mean_point
    #adjust SE of ATE for sample size
    cate_ate_df['ATE_stderr']=ate_inf.stderr_mean/len(xtest1)
    #conduct a Z test
    results=z_test(cate_ate_df['point_estimate'], cate_ate_df['ATE'], cate_ate_df['stderr'], cate_ate_df['ATE_stderr'])
    #also return which observations are significant 
    sig=results.abs()>=1.64
    return results, sig

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

def all_pdp_plots(xaxes, yaxes, model):
    '''
    Wrapper that plots PDPs for all variables into one large figure. 
    
    *vars=list of variables to be included in PDP plot 
    *spec=fitDML object which has x- and y-axis 
    *model=str; which model should be plotted, must be 'linear', 'cf' or ''
    '''
    #first get names of variables from x-axis dictionary
    variables=xaxes.keys()
    #define dimensions of figure - how many cols, how many rows 
    #want 4 columns 
    getcols=lambda x: len(x) if len(x)<=3 else 3
    n_cols=getcols(variables)
    getrows=lambda x, y: int(len(y)/x) if len(y)%x==0 else int(len(y)/x)+1
    n_rows=getrows(n_cols, variables)
    #set up figure
    fig, axes=plt.subplots(nrows=n_rows, ncols=n_cols, sharey=True, figsize=[30, 10])
    #flatten axes array to make looping easy if more than 1 row
    if n_rows>1: 
        axes=axes.flatten()
    #delete unnecessary axes 
    [fig.delaxes(axes[-(i+1)]) for i in range(n_cols*n_rows-len(variables))]
    #set colors and labels of CI and ATE - same in each axis 
    colors=['red', 'blue', 'red']    
    labels=['Lower CI', 'Point Estimate', 'Upper CI']
    #then plot pdp for each variable
    for i, var in enumerate(variables): 
        #get x- and y-axis 
        x_axis=xaxes[var]
        y_axis=yaxes[model][var]
        #plot these on axis[i] (only plot ATE for now)
        if n_cols>1:
            ax=axes[i]
        else: 
            ax=axes
        [ax.plot(x_axis, y_axis[:, j], color=col, label=lab) for j, col, lab in zip(range(y_axis.shape[1]-2), colors, labels)]
        #set title etc 
        ax.set_title(var)
    #set global legend 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='center left')
    #set global title and y-axis title (not yet working with my matplotlib version)
    fig.suptitle(f'Partial Dependence Plots of {model} model')
    fig.supylabel('MPC')
    figname='PDPs_'+model
    plt.savefig(fig_out/'PDP'/figname)
    plt.show()

def cdf_figure(spec, models, figname): 
    '''
    Plot empirical CDF with confidence interval bounds for all estimated models in specification.
    spec=fitDML object
    
    '''
    #calculate cdfs of each model
    for mod in models: 
        spec.get_cdfs(model=mod)
    #set up figure with one column for each model
    fig, axes=plt.subplots(nrows=1, ncols=len(models))
    #set colors (order: pe, upper ci, lower ci)
    colors=['blue', 'red', 'red']
    labels=['Point Estimate', 'upper CI', 'lower CI']
    #on each axis plot cdf of one model
    for ax, mod in zip(axes, models):
        #get cdf dict of model
        cdf=spec.cdfs[mod]
        #plot cdf of model on axis
        ax.plot(cdf['point_estimate'][0], cdf['point_estimate'][1], color=colors[0], label=labels[0])
        ax.plot(cdf['ci_upper'][0], cdf['ci_upper'][1], color=colors[1], label=labels[1])
        ax.plot(cdf['ci_lower'][0], cdf['ci_lower'][1], color=colors[2], label=labels[2])
    #set global legend 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
    #save figure
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
for col in hyperparams.columns[1:]: 
    hyperparams.loc[0, col]=int(hyperparams.loc[0, col])
    hyperparams.loc[1, col]=int(hyperparams.loc[1, col])
hyperparams=hyperparams.set_index('param', drop=True)
print('Hyperparameters loaded')
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
#get opt parameters for first stage for setting
best_params_R, best_params_Y=retrieve_params(hyperparams, treatment, outcome, 'spec1')
print('Spec 1 is set up.')

#* Estimation: Linear
#fit linear model 
folds=5
spec1_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec1_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

print('Spec 1 done')

#*#########################
#! Specification 2: Add Marriage 
#*#########################
#spec 2 uses only spec 1 variables plus marriage status 

#! FULL SAMPLE
#* Setup
#choose x_cols and set wcols
spec2_xcols=spec1_xcols+['married']
spec2_wcols=spec1_wcols
#only keep relevant variables 
suebset=variables[variables['comp_samp']==1]
spec2_df=subset[['custid']+[treatment, outcome]+spec2_xcols+spec2_wcols+constants]
#set up DML class
spec2_est=fitDML(spec2_df, treatment, outcome, spec2_xcols)
#get opt parameters for first stage for setting
best_params_R, best_params_Y=retrieve_params(hyperparams, treatment, outcome, 'spec2')
print('Spec 2 is set up.')

#* Estimation: Linear
#fit linear model 
folds=5
spec2_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec2_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

print('Spec 2 done')

#*#########################
#! Specification 3: Liquidity and salary
#*#########################
#add liqudity and salary variables
#! FULL SAMPLE
#* Setup
#choose x_cols 
spec3_xcols=['AGE', 'liqassii', 'married', 'FINCBTXM', 'FSALARYM']
spec3_wcols=spec2_wcols
#only keep relevant variables 
subset=variables[variables['l_samp']==1]
spec3_df=subset[['custid']+[treatment, outcome]+spec3_xcols+spec3_wcols+constants]
#set up DML class
spec3_est=fitDML(spec3_df, treatment, outcome, spec3_xcols)
#get opt parameters for first stage for setting
best_params_R, best_params_Y=retrieve_params(hyperparams, treatment, outcome, 'spec3')
print('Spec 3 is set up.')

#* Estimation: Linear
#fit linear model 
folds=5
spec3_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec3_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

print('Spec 3 done')

#*#########################
#! Specification 4: All financial variables
#*#########################
#spec 4 uses all financial variables (smallest sample available)

#! FULL SAMPLE
#* Setup
#choose x_cols 
spec4_xcols=spec3_xcols+['ORGMRTX', 'owned_m', 'notowned', 'QBLNCM1X']
spec4_wcols=spec3_wcols
#only keep relevant variables 
subset=variables[variables['comp_samp']==1]
spec4_df=subset[['custid']+[treatment, outcome]+spec4_xcols+spec4_wcols+constants]
spec4_df=spec4_df.dropna()
#set up DML class
spec4_est=fitDML(spec4_df, treatment, outcome, spec3_xcols)
#get opt parameters for first stage for setting
best_params_R, best_params_Y=retrieve_params(hyperparams, treatment, outcome, 'spec4')
print('Spec 4 is set up.')

#* Estimation: Linear
#fit linear model 
folds=5
spec4_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#* Estimation: Causal Forest 
#fit linear model 
folds=5
spec4_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)

print('Spec 4 done')

#*#########################
#! ANALYSIS
#*#########################

#*#########
#! Spec 1 
print('Start Spec 1')
#* CDF 
cdf_figure(spec=spec1_est, models=['linear', 'cf'], figname='cate_cdf_spec1')
#* PDP 
#get all axes for specification for both models and create PDPs
#linear model
spec1_est.all_pdp_axis(model='linear', alpha=0.1)
all_pdp_plots(spec1_est.x_axis, spec1_est.y_axis_pdp, model='linear')
#cf model 
spec1_est.all_pdp_axis(model='cf', alpha=0.1)
all_pdp_plots(spec1_est.x_axis, spec1_est.y_axis_pdp, model='cf')

#* Test ITE-ATE
testresults_lin=ite_ate_test(spec1_est, 'linear')
print(sum(testresults_lin[1]))
testresults_cf=ite_ate_test(spec1_est, 'cf')
print(sum(testresults_cf[1]))

print('Spec 1 done')

#*#########
#! Spec 2 
print('Start Spec 2')
#* CDF 
cdf_figure(spec=spec2_est, models=['linear', 'cf'], figname='cate_cdf_spec2')
#* PDP 
#get all axes for specification for both models and create PDPs
#linear model
spec2_est.all_pdp_axis(model='linear', alpha=0.1)
all_pdp_plots(spec2_est.x_axis, spec2_est.y_axis_pdp, model='linear')
#cf model 
spec2_est.all_pdp_axis(model='cf', alpha=0.1)
all_pdp_plots(spec2_est.x_axis, spec2_est.y_axis_pdp, model='cf')

#* Test ITE-ATE
testresults_lin=ite_ate_test(spec2_est, 'linear')
print(sum(testresults_lin[1]))
testresults_cf=ite_ate_test(spec2_est, 'cf')
print(sum(testresults_cf[1]))

print('Spec 2 done')

#*#########
#! Spec 3 
print('Start Spec 3')
#* CDF 
cdf_figure(spec=spec3_est, models=['linear', 'cf'], figname='cate_cdf_spec3')
#* PDP 
#get all axes for specification for both models and create PDPs
#linear model
spec3_est.all_pdp_axis(model='linear', alpha=0.1)
all_pdp_plots(spec3_est.x_axis, spec3_est.y_axis_pdp, model='linear')
#cf model 
spec3_est.all_pdp_axis(model='cf', alpha=0.1)
all_pdp_plots(spec3_est.x_axis, spec3_est.y_axis_pdp, model='cf')

#* Test ITE-ATE
testresults_lin=ite_ate_test(spec3_est, 'linear')
print(sum(testresults_lin[1]))
testresults_cf=ite_ate_test(spec3_est, 'cf')
print(sum(testresults_cf[1]))

print('Spec 3 done')

#*#########
#! Spec 4 
print('Start Spec 4')
#* CDF 
cdf_figure(spec=spec4_est, models=['linear', 'cf'], figname='cate_cdf_spec4')
#* PDP 
#get all axes for specification for both models and create PDPs
#linear model
spec4_est.all_pdp_axis(model='linear', alpha=0.1)
all_pdp_plots(spec4_est.x_axis, spec4_est.y_axis_pdp, model='linear')
#cf model 
spec4_est.all_pdp_axis(model='cf', alpha=0.1)
all_pdp_plots(spec4_est.x_axis, spec4_est.y_axis_pdp, model='cf')

#* Test ITE-ATE
testresults_lin=ite_ate_test(spec4_est, 'linear')
print(sum(testresults_lin[1]))
testresults_cf=ite_ate_test(spec4_est, 'cf')
print(sum(testresults_cf[1]))

print('Spec 4 done')