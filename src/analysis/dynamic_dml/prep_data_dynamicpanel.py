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

#* Panel Set 
#for dynamic DML need a balanced panel 
#keep only those in 2008
lvls_08=lvls_characteristics[lvls_characteristics['year']==2008]
#count how often each unit is observed 
obs_count=lvls_08[['custid', 'interviewno']].groupby(['custid']).count().rename(columns={'interviewno': 'count'})
#merge back onto lvls data and keep only those that have count=3
lvls_counts=lvls_08.merge(obs_count, on='custid')
lvls_count3_08=lvls_counts[lvls_counts['count']==3]

#* TESTING AREA RIGHT NOW
lvls_count3_08=lvls_count3_08.drop(['maritalstat', 'totbalance_ca', 'totbalance_sa'], axis=1)
lvls_count3_08=lvls_count3_08.replace(np.nan, 0)
y_train, y_test, t_train, t_test, z_train, z_test=du().create_test_train(lvls_count3_08, 'custid', 'TOTexp', 'RBTAMT', n_test=500)
x_train=z_train[['AGE', 'adults']]
x_test=z_test[['AGE', 'adults']]
w_train=z_train.drop(['AGE', 'adults'], axis=1)
w_test=z_test.drop(['AGE', 'adults'], axis=1)

n_panel=len(z_train['custid'].drop_duplicates())
n_periods=3
groups=np.repeat(a=np.arange(n_panel), repeats=n_periods, axis=0)
from econml.dynamic.dml import DynamicDML
from sklearn.ensemble import RandomForestRegressor as RFR
dyndml=DynamicDML(model_y=RFR(), model_t=RFR())
dyndml.fit(Y=y_train, T=t_train, X=x_train, W=w_train, groups=groups)
cme_inf=dyndml.const_marginal_effect_inference(X=x_test)
cme_inf_df=cme_inf.summary_frame()