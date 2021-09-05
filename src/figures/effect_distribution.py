import sys 
from pathlib import Path
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.distributions import empirical_distribution as ed

#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))

#* set data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'

'''
This file creates a figure that plots MPC against change in expenditure quantile.
'''
#*#########################
#! FUNCTIONS
#*#########################
def find_nearest(arr, val):
    arr=np.asarray(arr)
    idx=(np.abs(arr-val)).argmin()
    return idx, arr[idx]

def get_quantiles(arr, q=20):
    '''
    Get quantiles of empirical cdf.
    '''
    #prep input array
    arr=np.asarray(arr)
    arr_sort=np.sort(arr)
    #get ecdf
    ecdf=ed.ECDF(arr_sort)
    #find index of quantiles
    quants=np.linspace(0, 1, q+1)
    #index in ecdf is +1 of index in sorted array, hence substract 1
    #first value is associated qith 0 quantile, drop this
    q_idx=[find_nearest(ecdf.y, i)[0]-1 for i in quants][1:]
    #get values associated with quantiles
    q_vals=arr_sort[q_idx]
    
    return q_idx, q_vals

def quantile_mpc(arr, q, mpc):
    '''
    Find MPC of quantiles of arr.
    '''
    #prep arrays
    arr=np.asarray(arr)
    mpc=np.asarray(mpc)
    #get mpc of quantiles
    q_idx, q_vals=get_quantiles(arr=arr, q=q)
    #check where value in sorted array (returned by get_quantiles)
    #is equal to value in unsorted array
    arr_idx=[np.where(arr==i)[0][0] for i in q_vals]
    #then retrieve corresponding mpc out of mpc array 
    mpc_q=mpc[arr_idx]
    
    return mpc_q

#*#########################
#! DATA
#*#########################
#read in y, x and r data 
x_test=pd.read_csv(data_out/'transformed'/'x_test.csv')
x_test=x_test.drop(['Unnamed: 0'], axis=1)
y_test=pd.read_csv(data_out/'transformed'/'y_test.csv')
y_test=y_test.drop(['Unnamed: 0'], axis=1)
r_test=pd.read_csv(data_out/'transformed'/'r_test.csv')
#results 
mpc_spec1=pd.read_csv(data_out/'results'/'cate_spec1.csv')



#*#########################
#! PLOTS
#*#########################
#get mpc by quantile
mpc_q=quantile_mpc(y_test['chTOTexp'], 20, mpc_spec1['point_estimate'])
#
fig, ax=plt.subplots() 
ax.plot(np.linspace(0, 1, 21)[1:], mpc_q)