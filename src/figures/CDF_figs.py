from pathlib import Path 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'
figures=wd.parents[1]/'figures'

'''
This file creates figures using the marginal effects at the means estimated in demeaned_dml.py
'''

#*#########################
#! FUNCTIONS
#*#########################
def get_cdf(data): 
    '''
    Supply numpy array, pandas series or list of data and return proportional value and sorted values, i.e. CDF with x and y values.
    '''
    #point estimates sorted by size 
    pe_sorted=np.sort(np.array(data))
    #proportional values of sample 
    proportions=np.arange(len(pe_sorted))/(len(pe_sorted)-1)
    
    return(pe_sorted, proportions)

#*#########################
#! DATA
#*#########################
#MEAMs
meam_df=pd.read_csv(data_out/'results'/'MEAMS.csv')
#CMEs
cme_df=pd.read_csv(data_out/'results'/'CME.csv')
#X variables names
x_cols=['AGE_REF', 'children', 'QESCROWX', 'FSALARYM', 'FINCBTXM', 'adults', 'bothtop99pc', 'top99pc', 'MARITAL1_1']
#*Get CDF of MEAM 
#example now for AGE_REF 
cdf=get_cdf(meam_df['point_estimate_AGE_REF'])
#get the same for upper and lower bound of CI 
ci_lower_cdf=get_cdf(meam_df['ci_lower_AGE_REF'])
ci_upper_cdf=get_cdf(meam_df['ci_upper_AGE_REF'])
#*Get CDF of CME 
cdf_cme=get_cdf(cme_df['point_estimate'])
ci_lower=get_cdf(cme_df['ci_lower'])
ci_upper=get_cdf(cme_df['ci_upper'])

#*#########################
#! FIGURES
#*#########################
def cdf_figure(cdf, l_cdf, u_cdf): 
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
    
    return(fig, ax)

#!HOW TO VISUALIZE OUTLIERS?
cdf_figure(cdf_cme, ci_lower, ci_upper)

len(cme_df[cme_df['pvalue']<=0.1])