from pathlib import Path
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical 

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

#*#########################
#! FUNCTIONS
#*#########################
def trans_cats(series):
    '''
    Transform a pandas series containing a categorical variable categorized by strings into an integerbased categorization.
    '''
    #declare as categorical
    series_cat=pd.Categorical(series)
    #get codes
    series_catcodes=series_cat.cat.codes
    
    return series_catcodes

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
#for preliminary analysis, use the Parker et al. and Misra and Surico datasets
#!for now take a look at MS data as they have more variables 
ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/
                    '2008_data.csv')
ms_data=ms_data.dropna()
#list of observable characteristics in MS data - see 'MS_columnexplainers.csv' for definitions and notes on this 
observables=['AGE', 'AGE_REF', 'CKBK_CTX', 'CUTENURE', 'FAM_SIZE', 'FINCBTXM', 'FSALARYM', 'MARITAL1', 'NEWMRRT', 'ORGMRTX', 'QBLNCM1X', 'QESCROWX', 'SAVA_CTX', 'ST_HOUS', 'adults', 'bothtop99pc', 'chadults', 'chchildren', 'children', 'lastadults', 'lastchildren', 'lastage', 'lastrbtamt', 'validAssets', 'validIncome', 'timetrend', 'newid']
#dummy columns containing indices for time periods
dummies=['const']+['const'+str(num) for num in range(0, 15)]+['newid']
#variables related to rebate - see 'MS_columnexplainers.csv' for definitions and notes on this 
rebate_vars=['RBTAMT', 'RBTAMT_CHK', 'RBTAMT_EF', 'iREB', 'ifreb', 'ilreb', 'iontimereb_CHK', 'iontimereb_EF', 'ireballt', 'ireballt_CHK', 'ontimeRBTAMT', 'newid']
#create a DF with the observables,
X=ms_data[observables]
#...with rebate variables 
R=ms_data[rebate_vars]
#and one with month constants 
M=ms_data[dummies]

#*Categoricals 
#some observables are categoricals, which need to be encoded using OneHotEncoding
#-> computationally expensive, see https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree
#-> how to do it: https://stackoverflow.com/questions/24706677/how-to-handle-categorical-variables-in-sklearn-gradientboostingclassifier/24874515#24874515
#first create a list with categoricals and corresponding df
categoricals=['CKBK_CTX', 'MARITAL1', 'SAVA_CTX', 'ST_HOUS', 'timetrend']
#create df with categorical variables 
cats=ms_data[categoricals+['newid']]
#and create corresponding dummy df 
dummies_df=create_dummies(cats, id_col='newid')
#then drop categoricals raw from X df and merge dummies on it
X_nocats=X.drop(categoricals, axis=1)
X_w_dummies=X_nocats.merge(dummies_df, on='newid')

#*Observables dataframe 
#add constants to observables df
X_final=X_w_dummies.merge(M, on='newid')
#!create dummy variable 'married' that is 1 when married and 0 for all other statuses (divorced etc)
X_final['married']=X_final['MARITAL1']
#*Expenditures dataframe 
#get df with expenditure variables
expenditures=ms_data[[col for col in ms_data.columns if 'exp' in col]]

#*write to CSV 
X_final.to_csv(data_out/'transformed'/'MS_observables_dummies.csv')
R.to_csv(data_out/'transformed'/'MS_rebate_vars.csv')
expenditures.to_csv(data_out/'transformed'/'expenditures.csv')