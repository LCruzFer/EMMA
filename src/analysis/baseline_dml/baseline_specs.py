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
import matplotlib.ticker as plticker

#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils
from utils import fitDML
from ALEPython.src.alepython import ale_dml
#import ALEPython.src.alepython as ale_dml

#* set data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'
fig_out=wd.parents[2]/'figures'
results=wd.parents[2]/'data'/'results'

'''
This file estimates the baseline specifications of the linear DML model to estimate the constant marginal Conditional Average Treatment Effect (CATE), which is the MPC as a function of households characteristics.
'''

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

def subplots_canvas(variables): 
    #want 3 columns 
    getcols=lambda x: len(x) if len(x)<=3 else 3
    n_cols=getcols(variables)
    getrows=lambda x, y: int(len(y)/x) if len(y)%x==0 else int(len(y)/x)+1
    n_rows=getrows(n_cols, variables)
    #set up figure
    fig, axes=plt.subplots(nrows=n_rows, ncols=n_cols, figsize=[30, 10])
    #flatten axes array to make looping easy if more than 1 row
    if n_rows>1: 
        axes=axes.flatten()
    return fig, axes

def fig_geom(fig): 
    n_rows, n_cols=fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().get_geometry()
    return n_rows, n_cols

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
    figname='PDP_'+var+'_'+model
    plt.savefig(fig_out/'PDP'/figname)
    return fig, ax

def all_pdp_plots(xaxes, yaxes, model, spec):
    '''
    Wrapper that plots PDPs for all variables into one large figure. 
    
    *xaxes=dict; structure {var: x_axis values}
    *yaxes=dict of dicts; structure {model: {var: y_axis values}}
    *model=str; which model should be plotted, must be 'linear', 'cf' or ''
    *spec=str; specification used, needed for file name only
    '''
    #first get names of variables from x-axis dictionary
    variables=xaxes.keys()
    #define dimensions of figure - how many cols, how many rows 
    fig, axes=subplots_canvas(variables)
    n_rows, n_cols=fig_geom(fig)
    #delete unnecessary axes 
    [fig.delaxes(axes[-(i+1)]) for i in range(n_cols*n_rows-len(variables))]
    #set colors and labels of CI and ATE - same in each axis 
    colors=['red', 'blue', 'red']    
    labels=['Lower CI', 'Point Estimate', 'Upper CI']
    #then plot pdp for each variable
    for i, var in enumerate(variables): 
        print(var)
        #get x- and y-axis 
        x_axis=xaxes[var]
        y_axis=yaxes[model][var]
        #plot these on axis[i] (only plot ATE for now)
        if n_cols>1:
            ax=axes[i]
        else: 
            ax=axes
        #if binary variable, then use scatter 
        if (min(x_axis)==0)&(max(x_axis)==1): 
            [ax.scatter(x_axis, y_axis[:, j], color=col, label=lab) for j, col, lab in zip(range(y_axis.shape[1]-2), colors, labels)]
        #else plot 
        else: 
            [ax.plot(x_axis, y_axis[:, j], color=col, label=lab) for j, col, lab in zip(range(y_axis.shape[1]-2), colors, labels)]
        #set title etc 
        ax.set_title(var)
    #set global legend 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='center left')
    #set global title and y-axis title (not yet working with my matplotlib version)
    fig.suptitle(f'Partial Dependence Plots of {model} model')
    fig.supylabel('MPC')
    figname=spec+'PDPs_'+model
    plt.savefig(fig_out/'PDP'/outcome/figname)
    fig.clf()
    plt.close()

def all_ice_plots(xaxes, yaxes, model): 
    '''
    Wrapper function to create figure with ICE plots for each variable.
    
    *xaxes=dict; structure {var: x_axis values}
    *yaxes=dict of dicts; structure {model: {var: y_axis values}}
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
    fig, axes=plt.subplots(nrows=n_rows, ncols=n_cols, figsize=[30, 10])
    if n_rows>1:
        axes=axes.flatten()
    #for each variable plot all lines on one axis
    for i, var in enumerate(variables): 
        print(var)
        #get x-axis 
        x_axis=xaxes[var]
        #get y-axis lines 
        y_axis=yaxes[model][var]
        #get iˆth axis if n_col>1
        if n_cols>1: 
            ax=axes[i]
        else: 
            ax=axes
        #then plot each ICE of this variable on axis (only point estimate!)
        #if binary, use scatter 
        if (min(x_axis)==0)&(max(x_axis)==1):
            [ax.scatter(x_axis, y_axis[:, j, 1]) for j in range(y_axis.shape[1])]
        else: 
            [ax.plot(x_axis, y_axis[:, j, 1]) for j in range(y_axis.shape[1])]
        ax.set_title(var)
    #set global title and y-axis title (not yet working with my matplotlib version)
    fig.suptitle(f'Individual Expecations Plot of {model} model')
    fig.supylabel('MPC')
    figname='ICEs_'+model
    plt.savefig(fig_out/'ICE'/outcome/figname)
    fig.clf()
    plt.close()

def cdf_figure(spec, models, figname): 
    '''
    Plot empirical CDF with confidence interval bounds for all estimated models in specification.
    spec=fitDML object
    
    '''
    #calculate cdfs of each model
    for mod in models: 
        spec.get_cdfs(model=mod)
    #set up figure with one column for each model
    fig, axes=plt.subplots(nrows=1, 
                        ncols=len(models), 
                        figsize=(20, 10))
    #set colors (order: pe, upper ci, lower ci)
    colors=['limegreen', 'crimson', 'crimson']
    labels=['Point Estimate', 'upper CI', 'lower CI']
    #on each axis plot cdf of one model
    for ax, mod in zip(axes, models):
        #get cdf dict of model
        cdf=spec.cdfs[mod]
        #plot cdf of model on axis
        ax.plot(cdf['point_estimate'][0], 
                cdf['point_estimate'][1], 
                color=colors[0], label=labels[0], 
                linewidth=3, 
                )
        ax.plot(cdf['ci_upper'][0], 
                cdf['ci_upper'][1], 
                color=colors[1], label=labels[1], 
                linewidth=3, 
                )
        ax.plot(cdf['ci_lower'][0], 
                cdf['ci_lower'][1], 
                color=colors[2], label=labels[2], 
                linewidth=3, 
                )
        #set title of subplot
        ax.set_title(f'CDF of MPC using {mod} model')
        #set locations of ticks using tick locator 
        loc=plticker.MultipleLocator(base=0.5)
        ax.xaxis.set_major_locator(loc)
    #*global figure customization
    #rotate xaxis ticks globally 
    plt.xticks(rotation=45)
    #set global legend 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
    #set global title 
    fig.suptitle('CDF of point estimates')
    #save figure
    #plt.savefig(fig_out/'CDF'/outcome/figname)
    plt.show()
    fig.clf()
    plt.close()

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

def str_to_tex(file, tex_str):
    tex_file=open(results/'ate'/outcome/file, mode='a')
    tex_file.write(tex_str)
    tex_file.close()

def all_ale_plots(spec, model, bins=20,         
                bootstrap=False,
                bootstrap_samples=1000, n_sample=500, alpha=0.05, cut=1, 
                figname='ALE'): 
    '''
    Get single-feature ALE plots of predictor for all features in training set and save them in one figure.
    '''
    train_set=spec.x_test
    features=train_set.columns
    predictor=spec.selectmodel(model=model)
    #set up figure
    fig, axes=subplots_canvas(features)
    n_rows, n_cols=fig_geom(fig)
    #then for each feature
    for i, feat in enumerate(features): 
        print(feat)
    #!this is ugly so far
        if feat in ['married', 'owned_m', 'notowned']: 
            pass
        else:
            #get ALE, quantiles and CIs
            quants, ale, ci_low, ci_up, _=ale_dml.ale_bootstrap(predictor=predictor, train_set=train_set, feature=feat, bins=bins, bootstrap=bootstrap, bootstrap_samples=bootstrap_samples, n_sample=n_sample, alpha=alpha, cut=cut)
            #then create axes
            #get iˆth axis if n_col>1
            if n_cols>1: 
                ax=axes[i]
            else: 
                ax=axes
            #if bins are actually not existent, then its not possible to create the plot
            if (quants is None): 
                pass
            else:
                #plot 
                lines=[ci_low, ale, ci_up]
                labels=['lower CI', 'ALE', 'upper CI']
                colors=['red', 'green', 'red']
                for y, lab, col in zip(lines, labels, colors):
                    ax.plot(ale_dml._get_centres(quants), y, label=lab, color=col)
                    ax.set_title(feat)
    #set global legend, title and y_axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
    fig.suptitle(f'Partial Dependence Plots {model} model')
    fig.supylabel('ALE of MPC')
    #save figure
    plt.savefig(fig_out/'ALE'/outcome/figname)
    fig.clf()
    plt.close()

def do_analysis(spec, specname): 
    '''
    Apply all analysis steps to spec. 
    '''
    #* CDF 
    cdf_name='cdf_'+specname
    cdf_figure(spec=spec, models=['linear', 'cf'], figname=cdf_name)
    print('CDF done')
    #* PDP 
    # #get all axes for specification for both models and create PDPs
    # #linear model
    # spec.all_pdp_axis(model='linear', alpha=0.1)
    # all_pdp_plots(spec.x_axis_pdp, spec.y_axis_pdp, model='linear', spec=specname)
    # #cf model 
    # spec.all_pdp_axis(model='cf', alpha=0.1)
    # all_pdp_plots(spec.x_axis_pdp, spec.y_axis_pdp, model='cf', spec=specname)
    # print('PDP done')
    # # #* ICE 
    # #get all ICE axes for specifications 
    # #linear model
    # spec.all_ice_axis(model='linear')
    # all_ice_plots(spec.x_axis_ice, spec.y_axis_ice, model='linear')
    # #cf model 
    # spec.all_ice_axis(model='cf')
    # all_ice_plots(spec.x_axis_ice, spec.y_axis_ice, model='cf')
    #* ALE 
    #linear
    all_ale_plots(spec, model='linear', bins=20, bootstrap=True, bootstrap_samples=100, n_sample=spec.x_test.shape[0], figname=specname+'_linear')
    #cf
    all_ale_plots(spec, model='cf', bins=20, bootstrap=True, bootstrap_samples=100, n_sample=spec.x_test.shape[0], figname=specname+'_cf')
    print('ALE done')

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
#list of outcome variables
outcomes=[
        #!-> already done
        #'chTOTexp', 'chNDexp', 'chSNDexp', 'chFDexp',
        'chUTILexp', 'chVEHINSexp', 'chVEHFINexp']
#!spec 1 linear of chNDexp is still using iREB as treatment
#loop over all outcomes 
for out in outcomes:
    print(f'{out} start!')
    #! SET TREATMENT AND OUTCOME
    out='chNDexp'
    #set for all specifications
    #choose treatment
    treatment='RBTAMT'
    #choose outcome 
    outcome=out
    #set how many folds are done in second stage 
    folds=5
    #*#########################
    #! Specification 1: using MS specification
    #*#########################
    #specification 1 uses all available observations and hence only variables that have no missings 
    #!only for comparison reasons
    #* Setup
    #choose x and w columns
    spec1_xcols=['AGE', 'chFAM_SIZE']
    spec1_wcols=['AGE_sq', 'chFAM_SIZE_sq']
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
    spec1_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    #save marginal effect results in CSV 
    spec1_est.lin_cate_df.to_csv(results/outcome/'cate_spec1_lin.csv')
    #save ATE results in latex table 
    tex_str=spec1_est.lin_ate_inf.summary().as_latex()
    str_to_tex('spec1_lin_ate.tex', tex_str)   
    #* Estimation: Causal Forest 
    #fit causal forest model
    spec1_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    #save marginal effect results in CSV 
    spec1_est.cf_cate_df.to_csv(results/outcome/'cate_spec1_cf.csv')
    #save ATE results in latex table 
    tex_str=spec1_est.cf_ate_inf.summary().as_latex()
    str_to_tex('spec1_cf_ate.tex', tex_str)
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
    subset=variables[variables['comp_samp']==1]
    spec2_df=subset[['custid']+[treatment, outcome]+spec2_xcols+spec2_wcols+constants]
    #set up DML class
    spec2_est=fitDML(spec2_df, treatment, outcome, spec2_xcols)
    #get opt parameters for first stage for setting
    best_params_R, best_params_Y=retrieve_params(hyperparams, treatment, outcome, 'spec2')
    print('Spec 2 is set up.')

    #* Estimation: Linear
    #fit linear model 
    spec2_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    spec2_est.lin_cate_df.to_csv(results/outcome/'cate_spec2_lin.csv')
    tex_str=spec2_est.lin_ate_inf.summary().as_latex()
    str_to_tex('spec2_lin_ate.tex', tex_str)
    #* Estimation: Causal Forest 
    #fit causal forest model
    spec2_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    spec2_est.cf_cate_df.to_csv(results/outcome/'cate_spec2_cf.csv')
    tex_str=spec2_est.cf_ate_inf.summary().as_latex()
    str_to_tex('spec2_cf_ate.tex', tex_str)

    print('Spec 2 done')

    #*#########################
    #! Specification 3: Liquidity and salary
    #*#########################
    #add liqudity and salary variables
    #! FULL SAMPLE
    #* Setup
    #choose x_cols 
    spec3_xcols=['AGE', 'chFAM_SIZE', 'married', 'liqassii', 'FINCBTXM', 'FSALARYM']
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
    spec3_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    spec3_est.lin_cate_df.to_csv(results/outcome/'cate_spec3_lin.csv')
    tex_str=spec3_est.lin_ate_inf.summary().as_latex()
    str_to_tex('spec3_lin_ate.tex', tex_str)
    #* Estimation: Causal Forest 
    #fit cf model 
    spec3_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    spec3_est.cf_cate_df.to_csv(results/outcome/'cate_spec3_cf.csv')
    tex_str=spec3_est.cf_ate_inf.summary().as_latex()
    str_to_tex('spec3_cf_ate.tex', tex_str)

    print('Spec 3 done')

    #*#########################
    #! Specification 4: All financial variables
    #*#########################
    #spec 4 uses all financial variables (smallest sample available)
    #* Setup
    #choose x_cols 
    spec4_xcols=spec3_xcols+['ORGMRTX', 'owned_m', 'notowned', 'QBLNCM1X']
    spec4_wcols=spec3_wcols
    #only keep relevant variables 
    subset=variables[variables['comp_samp']==1]
    spec4_df=subset[['custid']+[treatment, outcome]+spec4_xcols+spec4_wcols+constants]
    spec4_df=spec4_df.dropna()
    #set up DML class
    spec4_est=fitDML(spec4_df, treatment, outcome, spec4_xcols)
    #get opt parameters for first stage for setting
    best_params_R, best_params_Y=retrieve_params(hyperparams, treatment, outcome, 'spec4')
    print('Spec 4 is set up.')

    #* Estimation: Linear
    #fit linear model 
    spec4_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    spec4_est.lin_cate_df.to_csv(results/outcome/'cate_spec4_lin.csv')
    tex_str=spec4_est.lin_ate_inf.summary().as_latex()
    str_to_tex('spec4_lin_ate.tex', tex_str)
    #* Estimation: Causal Forest 
    #fit cf model 
    spec4_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
    spec4_est.cf_cate_df.to_csv(results/outcome/'cate_spec4_cf.csv')
    tex_str=spec4_est.cf_ate_inf.summary().as_latex()
    str_to_tex('spec4_cf_ate.tex', tex_str)

    print('Spec 4 done')

    #*#########################
    #! ANALYSIS
    #*#########################

    #*#########
    #! Spec 1 
    print('Start Spec 1')
    do_analysis(spec1_est, 'spec1')
    # #* Test ITE-ATE
    # testresults_lin=ite_ate_test(spec1_est, 'linear')
    # print(sum(testresults_lin[1]))
    # testresults_cf=ite_ate_test(spec1_est, 'cf')
    # print(sum(testresults_cf[1]))
    print('Spec 1 done')

    #*#########
    #! Spec 2 
    print('Start Spec 2')
    do_analysis(spec2_est, 'spec2')
    # #* Test ITE-ATE
    # testresults_lin=ite_ate_test(spec2_est, 'linear')
    # print(sum(testresults_lin[1]))
    # testresults_cf=ite_ate_test(spec2_est, 'cf')
    # print(sum(testresults_cf[1]))
    print('Spec 2 done')

    #*#########
    #! Spec 3 
    print('Start Spec 3')
    do_analysis(spec3_est, 'spec3')
    # #* Test ITE-ATE
    # testresults_lin=ite_ate_test(spec3_est, 'linear')
    # print(sum(testresults_lin[1]))
    # testresults_cf=ite_ate_test(spec3_est, 'cf')
    # print(sum(testresults_cf[1]))
    print('Spec 3 done')

    #*#########
    #! Spec 4 
    print('Start Spec 4')
    do_analysis(spec4_est, 'spec4')
    # #* Test ITE-ATE
    # testresults_lin=ite_ate_test(spec4_est, 'linear')
    # print(sum(testresults_lin[1]))
    # testresults_cf=ite_ate_test(spec4_est, 'cf')
    # print(sum(testresults_cf[1]))
    print('Spec 4 done')
    print(f'{out} end!')