import sys
from pathlib import Path
import pandas as pd
import numpy as np

#* get data paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

'''
This file prepares the dataset for the dynamic DML estimation.
'''

#*#########################
#! FUNCTIONS
#*#########################
def reorder_df(columns, df): 
    '''
    Make columns the first columns of df in order they are contained in columns. 
    *columns=iterable containing column names 
    *df=pandas df to be reordered
    '''
    for num, col in enumerate(columns): 
        new_col=df.pop(col)
        df.insert(num, col, new_col)
    return df

def create_dummies(df, id_col):
    '''
    Transform all columns except id_col into type categorical.
    
    df=pandas.DataFrame with columns containing categorical variables 
    id_col=column with observation id that is not transformed, must be unique for each row
    '''
    #set id as index such that ignored in transformation
    df=df.set_index(id_col)
    #declare all columns as type categorical
    cat_df=df.astype('category')
    #create dummy df with pd.get_dummies() 
    dummy_df=pd.get_dummies(cat_df)
    #turn index into column again 
    dummy_df=dummy_df.reset_index()
    
    return dummy_df

def get_expnames(df): 
    '''
    Get column names of all expenditure variables, split into their level, change and lag versions.
    *df=pandas DataFrame
    '''
    #get columns that have 'exp' in their name
    exp_cols=[col for col in df.columns if 'exp' in col]
    #also filter between level, change and lag 
    lvl=[col for col in exp_cols if ('ch' not in col) & ('last' not in col)]
    ch=[col for col in exp_cols if 'ch' in col]
    lag=[col for col in exp_cols if 'last' in col]
    return lvl, ch, lag

def create_dummies(df, id_col):
    '''
    Transform all columns except id_col into type categorical.
    
    df=pandas.DataFrame with columns containing categorical variables 
    id_col=column with observation id that is not transformed, must be unique for each row
    '''
    #set id as index such that ignored in transformation
    df=df.set_index(id_col)
    #declare all columns as type categorical
    cat_df=df.astype('category')
    #create dummy df with pd.get_dummies() 
    dummy_df=pd.get_dummies(cat_df)
    #turn index into column again 
    dummy_df=dummy_df.reset_index()
    
    return dummy_df

#*#########################
#! DATA
#*#########################
#use Misra & Surico data
ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/'2008_data.csv')
#reorder this dataframe to make it easier to read 
ms_data=reorder_df(['custid', 'interviewno', 'newid', 'QINTRVMO', 'QINTRVYR'], ms_data)
#load parker et al data 
parkeretal=pd.read_stata(data_in/'Parkeretal_Data.dta')
parkeretal['newid']=[int(str(x)+str(y)) for x, y in zip(parkeretal['newid'], parkeretal['intview'])]
#merge some variables from Parker et al that are not included in MS 
ms_data=ms_data.merge(parkeretal[['newid', 'liqassii']], on='newid')
#replace NaN with 0 
ms_data=ms_data.replace(np.nan, 0)

#* General cleaning
#drop variables that are included twice - keep MS version because of better naming
nodoubles=ms_data.drop(['AGE_REF', 'PERSLT18'], axis=1)
#rename variables to names that are better understandable
newnames={'CKBK_CTX': 'totbalance_ca', 'FINCBTXM': 'tot_fam_inc', 
            'FSALARYM': 'fam_salary_inc', 'MARITAL1': 'maritalstat', 
            'SAVA_CTX': 'totbalance_sa', 'QINTRVMO': 'month', 
            'QINTRVYR': 'year', 'QBLNCM1X': 'out_princ'}
nodoubles=nodoubles.rename(columns=newnames)
#drop variables that are not expenditure, rebate, ID or characteristics 
#also drop ST_HOUS because all are ==2 (not student housing)
#validIncome and validAssets bear no useful information on their own (for now)
#!what is nmort?
cleaned=nodoubles.drop(['RESPSTAT', 'timetrend', 'dropconsumer',
                            'nmort', 'const', 'dropcust', 'timeleft', 
                            'lasttimetrend', 'chtimetrend', 'ST_HOUS', 
                            'validIncome', 'validAssets']
                            + ['const'+str(i) for i in range(15)], axis=1)
#also drop the rebate variables that have no meaning and the indicators 
#latter will be created if needed later on by myself again
#!maybe keep them later on if I know what those actually represent
cleaned=cleaned.drop(['iontimereb', 'ireballt', 'ireballt_CHK', 
                            'iREB', 'iontimereb_CHK', 'iontimereb_EF', 
                            'ilreb', 'ifreb', 'futrebamt'], axis=1)

#* Turn categoricals into two categories only 
#create dummies for categoricals that have several categories making a distinction between only two broader categories 
#married=1 if actually married, all other statuses =0
cleaned['married']=[int(x) for x in cleaned['maritalstat']==1]
#owned_nm=1 if housing owned w/o mortgage 
cleaned['owned_nm']=[int(x) for x in cleaned['CUTENURE']==2]
#owned_m=1 if housing owned w/ mortgage 
cleaned['owned_m']=[int(x) for x in cleaned['CUTENURE']==1]
#notowned=1 if housing not owned (all other CUTENURE categories)
cleaned['notowned']=[int(x) for x in (cleaned['CUTENURE']!=1)&(cleaned['CUTENURE']!=2)]

#* Create Dummies for Categoricals 
#first get subset of data that contains categoricals 
categoricals=['totbalance_ca', 'maritalstat', 'totbalance_sa', 'CUTENURE']
cats=cleaned[['newid']+categoricals]
#then create dummies for each of these 
cats_dummies=create_dummies(cats, 'newid')
#merge the dummies back to original data and drop the original variables 
cleaned=cleaned.drop(categoricals, axis=1)
cleaned_w_dummies=cleaned.merge(cats_dummies, on='newid')

#* Create quarter identifier 
#t dimension is quarter 
#need a running count from first quarter in data to last one 
cleaned_w_dummies['quarter']=1
for i in range(len(cleaned_w_dummies)): 
    year=cleaned_w_dummies.loc[i, 'year']
    month=cleaned_w_dummies.loc[i, 'month']
    if year==2008:
        if month in [1, 2, 3]: 
            cleaned_w_dummies.loc[i, 'quarter']=2
        elif month in [4, 5, 6]: 
            cleaned_w_dummies.loc[i, 'quarter']=3
        elif month in [7, 8, 9]: 
            cleaned_w_dummies.loc[i, 'quarter']=4
        elif month in [10, 11, 12]: 
            cleaned_w_dummies.loc[i, 'quarter']=5
    elif year==2009: 
        cleaned_w_dummies.loc[i, 'quarter']=6

#* Panel structure with lags
#count how often household is observed
count_obs=cleaned_w_dummies[['custid', 'newid']].groupby('custid').count().merge(cleaned_w_dummies[['custid', 'quarter']], on='custid').rename(columns={'newid': 'count'})
#can only use observations that have been observed at least 2 times 
cleaned_w_dummies=cleaned_w_dummies.merge(count_obs[['custid', 'count']], on='custid').drop_duplicates(['newid']).reset_index(drop=True)
panel=cleaned_w_dummies[cleaned_w_dummies['count']>1]
#lags 
#only need lag of treatments and outcomes 
lag_vars=[col for col in panel.columns if 'exp' in col]+[col for col in panel.columns if 'RBT' in col]

#then create lags
for var in lag_vars: 
    panel[var+'_lag']=panel[['custid', var]].groupby('custid').shift(-1)
#drop the 'last' variables 
panel_dropped=panel.drop([col for col in panel.columns if 'last' in col], axis=1)
#now only keep households that are only observed 3 times
panel_lags=panel_dropped[panel_dropped['count']==3]
#then remove rows that NaN in rows of lag
panel_nona_lags=panel_lags.dropna(subset=[col for col in panel_lags.columns if '_lag' in col])

#* Generate MS dataset (only containing their controls)
ms_data_reduced=ms_data[['custid', 'chTOTexp', 'RBTAMT','AGE', 'chchildren', 'chadults']+['const'+str(i) for i in range(0, 15)]]
ms_data_reduced['chFAM_SIZE']=ms_data_reduced['chchildren']+ms_data_reduced['chadults']
ms_data_reduced=ms_data_reduced.drop(['chchildren', 'chadults'], axis=1)
ms_data_reduced['sqAGE']=ms_data_reduced['AGE']**2
ms_data_reduced['sqchFAM_SIZE']=ms_data_reduced['chFAM_SIZE']**2

#* Write to CSV
ms_data_reduced.to_csv(data_out/'transformed'/'ms_setup.csv', index=False)
cleaned_w_dummies.to_csv(data_out/'transformed'/'cleaned_dummies.csv', index=False)
panel.to_csv(data_out/'transformed'/'panel_cleaned.csv', index=False)
panel_nona_lags.to_csv(data_out/'transformed'/'panel_w_lags.csv', index=False)
print('Finished!')