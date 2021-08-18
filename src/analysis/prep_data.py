import sys
from pathlib import Path
import pandas as pd
import numpy as np
#* set path to parent to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils as du
#* get data paths 
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'

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

#*#########################
#! DATA
#*#########################
#use Misra & Surico data
ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/'2008_data.csv')
#reorder this dataframe to make it easier to read 
ms_data=du().reorder_df(['custid', 'interviewno', 'newid', 'QINTRVMO', 'QINTRVYR'], ms_data)
#drop variables that are included twice - keep MS version because of bette naming
nodoubles=ms_data.drop(['AGE_REF', 'PERSLT18'], axis=1)
#drop changes and lags, will be calculated later on 
only_lvls=nodoubles[[col for col in nodoubles.columns if 'ch' not in col]]
only_lvls=only_lvls[[col for col in only_lvls.columns if 'last' not in col]]
#rename variables to names that are better understandable
newnames={'CKBK_CTX': 'totbalance_ca', 'FINCBTXM': 'tot_fam_inc', 
            'FSALARYM': 'fam_salary_inc', 'MARITAL1': 'maritalstat', 
            'SAVA_CTX': 'totbalance_sa', 'QINTRVMO': 'month', 
            'QINTRVYR': 'year'}
only_lvls=only_lvls.rename(columns=newnames)
#drop all variables that are not IDs, characteristics, rebate or expenditure related
#!what is nmort?
lvls_characteristics=only_lvls.drop(['nmort', 'RESPSTAT', 'timetrend',
                                    'dropcust', 'timeleft', 'const'] 
                                    +['const'+str(i) for i in range(15)], 
                                    axis=1)
