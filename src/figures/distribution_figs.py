import sys 
import os
from pathlib import Path
import numpy as np
from numpy.lib.shape_base import split 
import pandas as pd 
import matplotlib.pyplot as plt 

#* set data paths
wd=Path.cwd()
sys.path.append(str(wd.parent))
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'
fig_out=wd.parents[1]/'figures_new'
results=wd.parents[1]/'data'/'results_new'

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
    bins=np.linspace(mini, maxi+0.1, 40)
    return bins

def prep_data(variable, file):
    '''
    Read in and prep raw data.
    '''
    #for each axes, read in repsective dataframe 
    df=pd.read_csv(results/variable/file)
    #get dummy for significance
    df['significant']=(df['pvalue']<=0.1).astype(int)
    #get values
    values=df['point_estimate']
    #get bins and labels for bins        
    bins=get_bins(values)
    labels=np.linspace(1, len(bins), len(bins))
    #cut data into bins 
    df['bin']=pd.cut(values, bins=bins, labels=labels[:-1])
    
    return df, bins, labels

def get_shares(df):
    #get what percentage of values falls into bin i 
    shares=(df.groupby(['bin']).count()['point_estimate']/len(df))
    #get shares using only significant point estimates
    sig_shares=df.groupby(['bin']).sum()['significant']/df.groupby(['bin']).count()['point_estimate']
    sig_shares=sig_shares*shares
    
    return shares, sig_shares

def get_ate_xpos(ate, bins, labels):
    #return upper bound index 
    up_id=np.digitize(ate, bins)
    #get distance from lower bound to value as share of total length of interval -> relative position within bin
    rel_pos=(ate-bins[up_id-1])/(bins[up_id]-bins[up_id-1])
    #then get label values of bounds and add relative distance to lower bound -> absolute position on x-axis
    abs_pos=(labels[up_id-1])+rel_pos*(labels[up_id]-labels[up_id-1])
    return abs_pos

def split_filename(file):
    a=file.split('_')
    b=a[2].split('.')[0]
    file=a[1]+'_'+b
    return file

def get_title(file):
    '''
    Get title for figure from filename.
    '''
    split=file.split('_')[1:]
    #first get specification 
    spec=split[1]
    if spec=='spec1':
        specification='Specification 1'
    elif spec=='spec2':
        specification='Specification 2'
    elif spec=='spec3':
        specification='Specification 3'
    else:
        specification='Specification 4'
    #then get model type 
    mod=split[1]
    if mod=='lin.csv':
        model='Linear'
    else: 
        model='Causal Forest'
    title=f'{specification} using the \'{model}\' estimator'
    
    return title

def all_distributions_fig(variable):
    '''
    Specific function for my purpose - not generalized to create distribution figures (yet).
    '''
    #get list of result dfs
    files=os.listdir(results/variable)
    #kick anything that isn't a csv
    files=[file for file in files if '.csv' in file]
    files.sort()
    #set up figure 
    fig, axes=plt.subplots(nrows=4, ncols=2, figsize=[20, 15])
    #flatten axes into 1d array for easier looping
    axes=axes.flatten()
    for ax, file in zip(axes, files):
        #data prep
        df, bins, labels=prep_data(variable, file)
        #get shares 
        shares, sig_shares=get_shares(df)
        #*create barplots
        #add bars of shares to axis 
        ax.bar(labels[:-1], shares,
                color='teal' , edgecolor='black', 
                width=1, 
                align='edge', 
                linewidth=0.2, alpha=0.5)
        #add bars of significance shares 
        ax.bar(labels[:-1], sig_shares, 
                color='crimson', edgecolor='black',
                width=1, align='edge', 
                linewidth=0.2, 
                alpha=0.8)
        #add ATE as vertical dashed line 
        #retrieve ATE values
        ate=df['point_estimate'].mean()
        #get its position on x-axis 
        ate_pos=get_ate_xpos(ate, bins, labels)
        ax.axvline(ate_pos, 
                    linestyle='--', 
                    linewidth=0.5, 
                    color='black')
        #*axis customization
        #set x ticks and label them accordingly
        xticks=np.hstack((0, labels[:-1]))[::2]
        ax.set_xticks(xticks)
        xticklabels=np.round(bins, 2)[::2]
        #set xtick labels and rotate them 
        ax.set_xticklabels(xticklabels, 
                            rotation=45)
        #set extra tick at ATE value 
        #set y axis lim 
        ax.set_ylim((0, max(shares)+0.05))
        #set up title of axes
        title=file.split('_')[1]+' '+file.split('_')[2].split('.')[0]
        ax.set_title(f'Using {title} model')
    #*global figure customization
    #x-axis label
    fig.supxlabel('MPC', fontsize=15)
    #x-axis label
    #set global title
    fig.suptitle(f'Empirical Distribution of MPC with respect to {variable}')
    #distance between subplots
    fig.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.savefig(fig_out/'distributions'/variable,
                facecolor='white')

def distribution_fig(variable):
    '''
    Create single plots and save them for each estimator-specification pair.
    
    *variable=str; signals variable plots are created for.
    '''
    files=os.listdir(results/variable)
    #kick anything that isn't a csv and sort files
    files=[file for file in files if '.csv' in file]
    files.sort()
    for file in files:
        #create figure
        fig, ax=plt.subplots(figsize=[40, 30])
        #data prep
        df, bins, labels=prep_data(variable, file)
        #get shares 
        shares, sig_shares=get_shares(df)
        #*create barplots
        #add bars of shares to axis 
        ax.bar(labels[:-1], shares,
                color='teal' , edgecolor='black', 
                width=1, 
                align='edge', 
                linewidth=0.4, alpha=0.5)
        #add bars of significance shares 
        ax.bar(labels[:-1], sig_shares, 
                color='crimson', edgecolor='black',
                width=1, align='edge', 
                linewidth=0.4, 
                alpha=0.8)
        #add ATE as vertical dashed line 
        #retrieve ATE values
        ate=df['point_estimate'].mean()
        #get its position on x-axis 
        ate_pos=get_ate_xpos(ate, bins, labels)
        ax.axvline(ate_pos, 
                    linestyle='--', 
                    linewidth=4, 
                    color='black')
        #*axis customization
        #set x ticks and label them accordingly
        xticks=np.hstack((0, labels[:-1]))[::2]
        ax.set_xticks(xticks)
        xticklabels=np.round(bins, 2)[::2]
        #set xtick labels and rotate them 
        ax.set_xticklabels(xticklabels, 
                            rotation=45, 
                            fontsize=45)
        ax.tick_params(axis='y', labelsize=45, 
                        pad=20)
        #set extra tick at ATE value 
        #set y axis lim 
        ax.set_ylim((0, max(shares)+0.01))
        #set up title of axes
        #no title - use subfigure caption
        #ax.set_title(get_title(file))
        #x-axis label
        ax.set_xlabel('MPC', fontsize=55, 
                        labelpad=50)
        ax.set_ylabel('Frequency', fontsize=55, 
                        labelpad=50)
        #x-axis label
        #set global title
        #no title - use Latex caption
        #fig.suptitle(f'Empirical Distribution of MPC with respect to {variable}')
        #distance between subplots#
        fig.subplots_adjust(wspace=0.2, hspace=0.6)
        #save figure by getting filename and dropping 'cate_'
        filename=split_filename(file)+'_'+variable
        plt.savefig(fig_out/
                    'distributions_single'/filename,
                    facecolor='white')
        plt.close()

#*#########################
#! Create Figures
#*#########################
outcomes=[
        'chTOTexp', 'chNDexp', 'chSNDexp', 'chFDexp', 
        'chCARTKNexp', 'chPUBTRAexp', 'chAPPARexp', 'chHEALTHexp',
        'chUTILexp', 'chVEHINSexp', 'chENTERTexp'
        #! missing but not that relevant
        #'chVEHFINexp', 'chCARTKUexp', 
        ]
'''
for out in outcomes: 
    distribution_fig(out)
'''

for out in outcomes: 
    distribution_fig(out)