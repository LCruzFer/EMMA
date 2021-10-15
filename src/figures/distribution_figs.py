import sys 
import os
from pathlib import Path
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#* set data paths
wd=Path.cwd()
sys.path.append(str(wd.parent))
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'
fig_out=wd.parents[1]/'figures'
results=wd.parents[1]/'data'/'results'

#*#########################
#! FUNCTIONS
#*#########################
def get_min_max(x):
    minimum=np.min(x)
    minimum=np.divide(np.floor(minimum*10), 10)
    maximum=np.max(x)
    maximum=np.divide(np.ceil(maximum*10), 10)
    return minimum, maximum

def get_bins(x):
    mini, maxi=get_min_max(x)
    bins=np.linspace(mini, maxi+0.1, 20)
    return bins

def distribution_fig(variable):
    '''
    
    '''
    #get list of result dfs
    files=os.listdir(results/variable)
    files.sort()
    #set up figure 
    fig, axes=plt.subplots(nrows=4, ncols=2, figsize=[10, 10])
    axes=axes.flatten()
    for ax, file in zip(axes, files):
        #for each axes, read in a dataframe 
        df=pd.read_csv(results/variable/file)
        values=df['point_estimate']
        #get bins and labels for bins        
        bins=get_bins(values)
        labels=np.linspace(1, len(bins)-1, len(bins)-1)
        #cut data into bins 
        df['bin']=pd.cut(values, bins=bins, labels=labels)
        #get what percentage of values falls into bin i 
        shares=(df.groupby(['bin']).count()/len(df))['point_estimate']
        #add bars of shares to axis 
        ax.bar(labels, shares, width=1, align='edge', 
                linewidth=0.2, edgecolor='black')
        #set x ticks and label them accordingly
        xticks=np.hstack((0, labels))[::3]
        ax.set_xticks(xticks)
        xticklabels=np.round(bins[::3], 2)
        ax.set_xticklabels(xticklabels)
        title=file.split('_')[1]+' '+file.split('_')[2].split('.')[0]
        ax.set_title(f'{title}')
    fig.supxlabel('MPC')
    fig.suptitle(variable)
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(fig_out/'distributions'/variable)

#*#########################
#! Create Figures
#*#########################
outcomes=['chTOTexp', 'chNDexp', 'chSNDexp','chFDexp','chUTILexp', 'chVEHINSexp', 'chVEHFINexp']
for out in outcomes: 
    distribution_fig(out)