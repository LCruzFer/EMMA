import multiprocessing as mp 
import math
from multiprocessing import process
import time 
import sys
import itertools 
from pathlib import Path
from typing import AsyncGenerator
from numpy.lib.shape_base import column_stack
from numpy.typing import _128Bit
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#* set multiprocessing starting method to 'fork'
#forking - instead of spawning - creates a new process 
#and copies the main process it is 'forked' from
#with 'fork' don't have to do the whole __name__=='__main__' thing
mp.set_start_method('fork')

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

#*#########################
#! FUNCTIONS - V1
#*#########################
# def get_combis(l1, l2, l3):
#     combis=[i for i in itertools.product(l1, l2, l3)]
#     return combis

# def str_to_tex(file, tex_str, outcome):
#     tex_file=open(results/outcome/file, mode='a')
#     tex_file.write(tex_str)
#     tex_file.close()

# def retrieve_params(params_df, treatment, outcome, spec): 
#     '''
#     Get first stage parameters as dictionary for specification and setting.
#     *params_df=pandas df, output of tune_first_stage, index must be parameter name!
#     *treatment=str, treatment name 
#     *treatment=str, outcome name 
#     *spec=str, must be specX where X is number of specification
#     '''
#     treat_col=treatment+'_'+spec
#     spec_col=outcome+'_'+spec
#     params_T={param: val for param, val in zip(params_df.index, params_df[treat_col])}
#     params_Y={param: val for param, val in zip(params_df.index, params_df[spec_col])}
#     print(f'Got parameters for {outcome} in {spec}')
#     return params_T, params_Y

# def get_xwcols(spec):
#     ''' 
#     Ugly but quickly written function to retrieve cols - due to legacy error in how estimation class is structured.
#     '''
#     spec1_xcols=['AGE', 'chFAM_SIZE']
#     spec1_wcols=['AGE_sq', 'chFAM_SIZE_sq']
#     spec2_xcols=spec1_xcols+['married']
#     spec2_wcols=spec1_wcols
#     spec3_xcols=['AGE', 'chFAM_SIZE', 'married', 'liqassii', 'FINCBTXM', 'FSALARYM']
#     spec3_wcols=spec2_wcols
#     spec4_xcols=spec3_xcols+['ORGMRTX', 'owned_m', 'notowned', 'QBLNCM1X']
#     spec4_wcols=spec3_wcols
#     if spec=='spec1':
#         return spec1_xcols, spec1_wcols
#     if spec=='spec2':
#         return spec2_xcols, spec2_wcols
#     if spec=='spec3':
#         return spec3_xcols, spec3_wcols
#     if spec=='spec4':
#         return spec4_xcols, spec4_wcols

# def get_sample(spec, df, outcome, treatment): 
#     '''
#     Get sample and x-cols or spec. 
#     Ugly but quickly written.
#     '''
#     #get sample specific for this specification 
#     if (spec=='spec1')|(spec=='spec2'):
#         subset=df[df['comp_samp']==1]
#     elif spec=='spec3':
#         subset=df[df['l_samp']==1]
#     else: 
#         subset=df
#     spec_xcols, spec_wcols=get_xwcols(spec)
#     constants=['const'+str(i) for i in range(1, 15)]
#     out_df=subset[['custid', outcome, treatment]+spec_xcols+spec_wcols+constants]
#     if spec=='spec4':
#         out_df=out_df.dropna()
#     return out_df, spec_xcols

# def subplots_canvas(variables): 
#     #want 3 columns 
#     getcols=lambda x: len(x) if len(x)<=3 else 3
#     n_cols=getcols(variables)
#     getrows=lambda x, y: int(len(y)/x) if len(y)%x==0 else int(len(y)/x)+1
#     n_rows=getrows(n_cols, variables)
#     #set up figure
#     fig, axes=plt.subplots(nrows=n_rows, ncols=n_cols, figsize=[30, 10])
#     #flatten axes array to make looping easy if more than 1 row
#     if n_rows>1: 
#         axes=axes.flatten()
#     return fig, axes

# def fig_geom(fig): 
#     n_rows, n_cols=fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().get_geometry()
#     return n_rows, n_cols

# def ale_plot(predictor, train, features, outcome, modelspec):
#     '''
#     Make ALE plot for features using spec-model (linear or cf) combi.
#     '''
#     #set up figure
#     fig, axes=subplots_canvas(features)
#     n_rows, n_cols=fig_geom(fig)
#     #then for each feature
#     for i, feat in enumerate(features): 
#         print(feat)
#     #!this is ugly so far
#         if feat in ['married', 'owned_m', 'notowned']: 
#             pass
#         else:
#             #get ALE, quantiles and CIs
#             quants, ale, ci_low, ci_up, _=ale_dml.ale_bootstrap(predictor=predictor, train_set=train, feature=feat, bins=20, bootstrap=True, bootstrap_samples=200, n_sample=train.shape[0], alpha=0.1, cut=1)
#             #then create axes
#             #get iˆth axis if n_col>1
#             if n_cols>1: 
#                 ax=axes[i]
#             else: 
#                 ax=axes
#             #if bins are actually not existent, then its not possible to create the plot
#             if (quants is None): 
#                 pass
#             else:
#                 #plot 
#                 lines=[ci_low, ale, ci_up]
#                 labels=['lower CI', 'ALE', 'upper CI']
#                 colors=['red', 'green', 'red']
#                 for y, lab, col in zip(lines, labels, colors):
#                     ax.plot(ale_dml._get_centres(quants), y, label=lab, color=col)
#                     ax.set_title(feat)
#     #set global legend, title and y_axis
#     handles, labels=ax.get_legend_handles_labels()
#     fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
#     fig.suptitle(f'Partial Dependence Plots of {modelspec}')
#     fig.supylabel('ALE of MPC')
#     #save figure
#     plt.savefig(fig_out/'ALE'/outcome/modelspec)
#     plt.close()

# def cdf_figure(spec, models, outcome, figname): 
#     '''
#     Plot empirical CDF with confidence interval bounds for all estimated models in specification.
#     spec=fitDML object    
#     '''
#     #calculate cdfs of each model
#     for mod in models: 
#         spec.get_cdfs(model=mod)
#     #set up figure with one column for each model
#     fig, axes=plt.subplots(nrows=1, ncols=len(models), figsize=(10, 20))
#     #set colors (order: pe, upper ci, lower ci)
#     colors=['blue', 'red', 'red']
#     labels=['Point Estimate', 'upper CI', 'lower CI']
#     #on each axis plot cdf of one model
#     for ax, mod in zip(axes, models):
#         #get cdf dict of model
#         cdf=spec.cdfs[mod]
#         #plot cdf of model on axis
#         ax.plot(cdf['point_estimate'][0], cdf['point_estimate'][1], color=colors[0], label=labels[0])
#         ax.plot(cdf['ci_upper'][0], cdf['ci_upper'][1], color=colors[1], label=labels[1])
#         ax.plot(cdf['ci_lower'][0], cdf['ci_lower'][1], color=colors[2], label=labels[2])
#         ax.set_title(f'CDF of MPC using {mod} model')
#     #set global legend 
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
#     #save figure
#     plt.savefig(fig_out/'CDF'/outcome/figname)
#     fig.clf()
#     plt.close()

# def do_estimation(treatment, 
#                     outcome, 
#                     spec, 
#                     data, 
#                     params_df, 
#                     folds=5):
#     '''
#     Run the whole estimation of treatment on outcome using specification spec.
#     '''
#     print(f'Start estimation for {treatment, outcome, spec}')
#     #get parameters 
#     best_params_R, best_params_Y=retrieve_params(params_df, treatment, outcome, spec)
#     #get df for specification 
#     spec_df, spec_xcols=get_sample(spec=spec, df=data, outcome=outcome, treatment=treatment)
#     #set up estimation class
#     spec_est=fitDML(spec_df, treatment, outcome, spec_xcols)
#     print(f'{spec} for {outcome} on {treatment} is set up.')
#     #fit linear model for specification
#     spec_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#     #then write to csv 
#     filename='lin_cate_'+spec+'.csv'
#     spec_est.lin_cate_df.to_csv(results/outcome/filename)
#     #and ate to latex
#     tex_str=spec_est.lin_ate_inf.summary().as_latex()
#     filename='lin_ate_'+spec+'.tex'
#     str_to_tex(filename, tex_str, outcome)
#     #get ALE plots for each feature in X
#     ale_plot(predictor=spec_est.linDML, train=spec_est.x_train, features=spec_est.x_cols, outcome=outcome, modelspec=spec+'_linear')
#     print(f'linear done for {treatment, outcome, spec}')
#     #fit cf model for specification 
#     spec_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#     #then write to csv 
#     filename='cf_cate_'+spec+'.csv'
#     spec_est.lin_cate_df.to_csv(results/outcome/filename)
#     #and ate to latex
#     tex_str=spec_est.lin_ate_inf.summary().as_latex()
#     filename='cf_ate_'+spec+'.tex'
#     str_to_tex(filename, tex_str, outcome)
#     #get ALE plots
#     ale_plot(predictor=spec_est.cfDML, train=spec_est.x_train, features=spec_est.x_cols, outcome=outcome, modelspec=spec+'_cf')
#     print(f'cf done for {treatment, outcome, spec}')
#     #get CDF for both models 
#     cdf_figure(spec_est, models=['linear', 'cf'], outcome=outcome, figname=spec+'_CDFs')

#*#########################
#! DATA - V1
#*#########################
# #read in data
# variables=pd.read_csv(data_out/'transformed'/'prepped_data.csv')
# print('Variables loaded')

# #* Random Forest Hyperparameters
# #read in hyperparameters for RF - output from tune_first_stage.py
# hyperparams=pd.read_csv(data_out/'transformed'/'first_stage_hyperparameters.csv')
# #rename hyperparams column 
# hyperparams=hyperparams.rename(columns={'Unnamed: 0': 'param'})
# #need to turn some entries into integers that are read in str 
# for col in hyperparams.columns[1:]: 
#     hyperparams.loc[0, col]=int(hyperparams.loc[0, col])
#     hyperparams.loc[1, col]=int(hyperparams.loc[1, col])
# hyperparams=hyperparams.set_index('param', drop=True)
# print('Hyperparameters loaded')

# #get combinations 
# outcomes=[
#         'chFDexp', 
#         # 'chTOTexp', 
#         # 'chNDexp', 
#         # 'chSNDexp'
#         ]
# treatment=['RBTAMT']
# specs=[
#     'spec1', 
#     # 'spec2', 
#     # 'spec3', 
#     # 'spec4'
#     ]
# combis=get_combis(treatment, outcomes, specs)

# #*#########################
# #! ESTIMATION - V1
# #*#########################
# start=time.perf_counter()
# processes=[]
# for combi in combis:
#     treat=combi[0]
#     out=combi[1]
#     spec=combi[2]
#     processes.append(mp.Process(target=do_estimation, args=[treat, out, spec, variables, hyperparams]))
#     print(f'process for {combi} set up.')
# for p in processes: 
#     p.start() 
# for p in processes: 
#     p.join()
# # with concurrent.futures.ProcessPoolExecutor() as executor: 
# #     f=[executor.submit(do_estimation, args=[combi[0], combi[1], combi[2], variables, hyperparams]) for combi in combis]
# finish=time.perf_counter()
# print(f'The processes were finished in {finish-start} seconds.')

#*#########################
#! FUNCTIONS - V2
#*#########################
#here using multiprocessing for ALE bootstraps 

def cdf_figure(spec, models, figname, outcome): 
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
        ax.set_title(f'CDF of MPC using {mod} model')
    #set global legend 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
    #save figure
    #plt.savefig(fig_out/'CDF'/outcome/figname)
    plt.show()
    fig.clf()
    plt.close()

def get_combis(l1, l2, l3):
    combis=[i for i in itertools.product(l1, l2, l3)]
    return combis

def str_to_tex(file, tex_str, outcome):
    tex_file=open(results/outcome/file, mode='a')
    tex_file.write(tex_str)
    tex_file.close()

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
    print(f'Got parameters for {outcome} in {spec}')
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

def all_ale_plots(spec, model, outcome,
                bins=20,         
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
    #!this is ugly so far
        if feat in ['married', 'owned_m', 'notowned', 'chFAM_SIZE']: 
            pass
        else:
            #get ALE, quantiles and CIs
            quants, ale, ci_low, ci_up, _=ale_dml.ale_mp_bootstrap(predictor=predictor, train_set=train_set, feature=feat, bins=bins, bootstrap=bootstrap, bootstrap_samples=bootstrap_samples, n_sample=n_sample, alpha=alpha, cut=cut)
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
        print(feat)
    #set global legend, title and y_axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1,0.5), loc='upper left')
    fig.suptitle(f'Partial Dependence Plots {model} model')
    fig.supylabel('ALE of MPC')
    #save figure
#    plt.savefig(fig_out/'ALE'/outcome/figname)
    plt.show()
    fig.clf()
    plt.close()

def do_analysis(spec, specname, outcome): 
    '''
    Apply all analysis steps to spec. 
    '''
    #* CDF 
    cdf_name='cdf_'+specname
    cdf_figure(spec=spec, models=['linear', 'cf'], figname=cdf_name, outcome=outcome)
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
    all_ale_plots(spec, model='linear', outcome=outcome, bins=20, bootstrap=True, bootstrap_samples=100, n_sample=spec.x_test.shape[0], figname=specname+'_linear')
    #cf
    all_ale_plots(spec, model='cf', outcome=outcome, bins=20, bootstrap=True, bootstrap_samples=100, n_sample=spec.x_test.shape[0], figname=specname+'_cf')
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

#get combinations 
outcomes=[
        # 'chFDexp', 
        # 'chTOTexp', 
        # 'chNDexp', 
        # 'chSNDexp', 
        'chUTILexp', 
        'chVEHINSexp', 
        'chVEHFINexp'
        ]
treatment=['RBTAMT']
specs=[
    'spec1', 
    'spec2', 
    'spec3', 
    'spec4'
    ]
combis=get_combis(treatment, outcomes, specs)

#*#########################
#! ESTIMATION - V2
#*#########################
out='chNDexp'
#!the following is taken from baseline_dml.py for testing purposes
treatment='RBTAMT'
#choose outcome 
outcome=out
#set how many folds are done in second stage 
folds=5
constants=['const'+str(i) for i in range(1, 15)]
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

start=time.perf_counter()
#* Estimation: Linear
#fit linear model 
spec1_est.fit_linear(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#save marginal effect results in CSV 
#spec1_est.lin_cate_df.to_csv(results/outcome/'cate_spec1_lin.csv')
#save ATE results in latex table 
tex_str=spec1_est.lin_ate_inf.summary().as_latex()
#str_to_tex('spec1_lin_ate.tex', tex_str)   
#* Estimation: Causal Forest 
#fit causal forest model
spec1_est.fit_cfDML(params_Y=best_params_Y, params_T=best_params_R, folds=folds)
#save marginal effect results in CSV 
#spec1_est.cf_cate_df.to_csv(results/outcome/'cate_spec1_cf.csv')
#save ATE results in latex table 
tex_str=spec1_est.cf_ate_inf.summary().as_latex()
#str_to_tex('spec1_cf_ate.tex', tex_str)
print('Estimation done')
print('Start Spec 1')
do_analysis(spec1_est, 'spec1', outcome=outcome)
print('analysis done')
finish=time.perf_counter()
print(f'Esimtation and analysis took {finish}-{start} seconds')