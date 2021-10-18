import sys 
from pathlib import Path 
import pandas as pd 
import numpy as np
from scipy.sparse.base import spmatrix
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#* set system path to import utils
wd=Path.cwd()
#* set data paths
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'
fig_out=wd.parents[1]/'figures'

#*#########################
#! FUNCTIONS
#*#########################
def group_plot(xaxis, yaxis, groups, xlab='Month', ylab='Liquidity', title='Changes in liquidity within groups'): 
    '''
    Plot y axis for different groups in same plot.
    '''
    fig, ax=plt.subplots()
    for i in groups:
        ax.plot(xaxis, yaxis[i], label=f'Group {i+1}')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    return fig, ax

#*#########################
#! DATA
#*#########################
#read in raw data for first quarter of 2008
#additionally second quarter to compare to April (buffer time)
raw081=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli081x.csv')
raw082=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli082.csv')
raw=raw081.append(raw082)
#get interview number 
raw['intrvwno']=raw['NEWID'].apply(lambda x: int(str(x)[-1]))
#only use households that have not yet received their rebate (!)
#load rebate data 
rbt08=pd.read_csv(data_in/'2008_rawdata'/'expn08'/'rbt08.csv')
#get IDs of households that already received their rebate
rbtrcvd=rbt08[rbt08['RBTMO']<5]['NEWID']
#only keep relevant variables that we have in main data 
variables=['NEWID', 'intrvwno', 'QINTRVYR', 'QINTRVMO', 'AGE_REF', 'SAVACCTX', 'MARITAL1', 'CKBKACTX', 'FINCBTXM', 'FSALARYM', 'CUTENURE', 'ST_HOUS', 'PERSLT18', 'AGE2', 'FAM_SIZE']
#not in fmli: 'QESCROWX', 'QMRTTERM', 'ORGMRTX', 'QBLNCM1X', 'NEWMRRT'
raw_vars=raw[variables]
#drop irrelevant months 
months=[1, 2, 3, 4]
relevant=raw_vars[raw_vars['QINTRVMO'].isin(months)]
#and only keep households that have not yet received rebate
relevant=relevant[relevant['NEWID'].isin(rbtrcvd)==False]
#drop rows that have NA 
relevant=relevant.replace('.', np.nan)
relevant['SAVACCTX']=relevant['SAVACCTX'].astype(float)
relevant['CKBKACTX']=relevant['CKBKACTX'].astype(float)
#also AGE2 is a string
relevant['AGE2']=relevant['AGE2'].astype(float)
relevant['AGE_REF']=relevant['AGE_REF'].astype(float)
#AGE=mean(AGE+AGE2) if AGE2!=np.nan or 0 
relevant['AGE']=relevant['AGE_REF']
relevant['AGE'][(relevant['AGE2']!=np.nan)|(relevant['AGE2']!=0)]=(relevant['AGE2']+relevant['AGE_REF'])/2
relevant=relevant.drop(['AGE2', 'AGE_REF'], axis=1)
#create liquidity variable: if one of the reported is NaN make sure that liquidity=other variable 
relevant['liqassii']=relevant['SAVACCTX'].add(relevant['CKBKACTX'], fill_value=0)
relevant[['liqassii', 'SAVACCTX', 'CKBKACTX']]
relev_nona=relevant.dropna()
#for grouping ignore liquidity, year is 2008 anyway and nor relevant 
df=relev_nona.drop(['SAVACCTX', 'CKBKACTX', 'QINTRVYR'], axis=1).set_index(['NEWID', 'QINTRVMO'], drop=True)
liquid_df=df[['liqassii']]
df=df.drop(['liqassii'], axis=1)
#scale everything 
scaler=StandardScaler()
scaled=scaler.fit_transform(df)

#*#########################
#! ANALYSIS
#*#########################
#* Apply k means algorithm
#set number of clusters
n_clust=5
#set up Kmeans
kmeans=KMeans(init='random', n_clusters=n_clust, n_init=10, max_iter=300, random_state=42)
#fit K-means algorithm 
kmeans.fit(scaled)
#get group assignment for each observation
liquid_df['group']=kmeans.labels_
#get group-month averages of liquidity
liquid_df=liquid_df.reset_index(level='QINTRVMO')
liqavg=liquid_df.groupby(['group', 'QINTRVMO']).mean().reset_index()
liqmedian=liquid_df.groupby(['group', 'QINTRVMO']).median().reset_index()
liqavg=liqavg.merge(liqmedian, on=['group', 'QINTRVMO'])
liqavg['pre_announce']=((liqavg['QINTRVMO']==1)|(liqavg['QINTRVMO']==2)).astype(int)
liqavg2=liqavg[['group', 'pre_announce', 'liqassii']].groupby(['group', 'pre_announce']).mean()
liqavg2=liqavg2.reset_index()

#* Plot 
#get array with group numbers
groups=np.linspace(start=0, stop=n_clust-1, num=n_clust, dtype=int)
#set months for x axis
months=[1, 2, 3, 4]
x_axis=np.array(months)
#then get y_axis 
y_axis=np.array([liqavg['liqassii_x'][liqavg['group']==i] for i in groups])
#then plot group
group_plot(x_axis, y_axis, groups=groups)
