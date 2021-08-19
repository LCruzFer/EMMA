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

#* General cleaning
#drop variables that are included twice - keep MS version because of better naming
nodoubles=ms_data.drop(['AGE_REF', 'PERSLT18'], axis=1)
#rename variables to names that are better understandable
newnames={'CKBK_CTX': 'totbalance_ca', 'FINCBTXM': 'tot_fam_inc', 
            'FSALARYM': 'fam_salary_inc', 'MARITAL1': 'maritalstat', 
            'SAVA_CTX': 'totbalance_sa', 'QINTRVMO': 'month', 
            'QINTRVYR': 'year'}
nodoubles=nodoubles.rename(columns=newnames)
#drop variables that are not expenditure, rebate, ID or characteristics 
#!what is nmort?
cleaned=nodoubles.drop(['RESPSTAT', 'timetrend', 'dropconsumer',
                            'nmort', 'const', 'dropcust', 'timeleft', 
                            'lasttimetrend', 'chtimetrend']
                            + ['const'+str(i) for i in range(15)], axis=1)
#also drop the rebate variables that have no meaning and the indicators 
#latter will be created if needed later on by myself again
#!maybe keep them later on if I know what those actually represent
cleaned=cleaned.drop(['iontimereb', 'ireballt', 'ireballt_CHK', 
                            'iREB', 'iontimereb_CHK', 'iontimereb_EF', 
                            'ilreb', 'ifreb', 'futrebamt'], axis=1)

#* Create Dummies for Categoricals 
#first get subset of data that contains categoricals 
categoricals=['totbalance_ca', 'maritalstat', 'totbalance_sa', 'ST_HOUS']
cats=cleaned[['newid']+categoricals]
#then create dummies for each of these 
cats_dummies=create_dummies(cats, 'newid')
#merge the dummies back to original data and drop the original variables 
cleaned=cleaned.drop(categoricals, axis=1)
cleaned_w_dummies=cleaned.merge(cats_dummies, on='newid')

#write this version to CSV
cleaned_w_dummies.to_csv(data_out/'transformed'/'cleaned_dummies.csv', index=False)