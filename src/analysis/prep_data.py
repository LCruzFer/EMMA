import sys 
from pathlib import Path 
import pandas as pd 
import numpy as np 

#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))

#* set data paths
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

#*#########################
#! FUNCTIONS
#*#########################
def gen_samp_dummy(df, var, idvar='newid'): 
    '''
    Generate a dummy that signals whether variable is in sample where var is not NAN.
    
    *df=pd DataFrame 
    *var=str; column name in df on which basis sample is generated
    *idvar=str; column name in df that signals id 
    '''
    dummies=(df[idvar].isin(df[df[var].isnull()==False][idvar])).astype(int)
    return dummies

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
#read in data 
ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/'2008_data.csv')
#also read in csv for sample dummies
parker=pd.read_stata(data_in/'Parkeretal_Data.dta')
parker['newid']=(parker['newid'].astype(str)+parker['intview'].astype(str)).astype(int)
#then merge 
ms_data=ms_data.merge(parker[['newid', 'samplemain', 'sample2', 'sample2a', 'sample3', 'sample4', 'sample', 'sample4l2i']], on='newid', how='left')
#age and children are included twice - keep MS version 
ms_data=ms_data.drop(['AGE_REF', 'PERSLT18'], axis=1)
#load parker et al data for the liquid assets data 
parker=pd.read_stata(data_in/'Parkeretal_Data.dta')
parker['newid']=(parker['newid'].astype(str)+parker['intview'].astype(str)).astype(int)
ms_data=ms_data.merge(parker[['newid', 'liqassii']], on='newid')

#! get some more variables from raw data potentially 
#read in relevant fmli files and merge into one 
rbt08=pd.read_csv(data_in/'2008_rawdata'/'expn08'/'rbt08.csv')
rbt08['custid']=rbt08['NEWID'].apply(lambda x: int(str(x)[:-1]))
rbt08=rbt08.sort_values('RBTMO')
rbt08=rbt08.drop_duplicates(subset='custid', keep='first')
ms_data=ms_data.merge(rbt08[['custid', 'RBTMO']], on='custid', how='left')


#* Variable Lists
#most important observables
observables=['AGE', 'SAVA_CTX', 'MARITAL1', 'CKBK_CTX', 'QESCROWX', 'QMRTTERM', 'ORGMRTX', 'FINCBTXM', 'FSALARYM', 'liqassii', 'adults', 'children', 'FAM_SIZE', 'QBLNCM1X', 'NEWMRRT', 'CUTENURE', 'ST_HOUS']
#ID and indicator variables 
meta=['custid', 'newid', 'QINTRVYR', 'QINTRVMO', 'interviewno']
#treatment variables 
treatment=['RBTAMT', 'RBTAMT_CHK', 'RBTAMT_EF', 'ontime', 'iREB', 'ontimeRBTAMT', 'ireballt', 'ireballt_CHK', 'iontimereb', 'iontimereb_CHK', 'iontimereb_EF', 'lastrbtamt', 'futrebamt', 'ilreb', 'ifreb']
#dependent variables (change in expenditure)
chexpvars=[col for col in ms_data.columns if ('exp' in col)&('ch' in col)]
#lagged variables
lastvars=[col for col in ms_data.columns if 'last' in col]
#contemporaneous expenditure 
expvars=[col for col in ms_data.columns if ('exp' in col)&('ch' not in col)&('last' not in col)]
#change in observables
chvars=[col for col in ms_data.columns if ('exp' not in col)&('ch' in col)]
chvars.remove('children')
chvars.remove('lastchildren')
#timetrends 
timevars=['timetrend', 'const']+['const'+str(i) for i in range(0, 15)]
#samplevars
samplevars=['samplemain', 'sample2', 'sample2a', 'sample3', 'sample4', 'sample', 'sample4l2i']
#other variables 
othervars=[col for col in ms_data.columns if col not in observables+meta+treatment+chexpvars+lastvars+chvars+expvars+timevars+samplevars]

#* Dropping Zone
#drop othervars - not of interest 
ms_data=ms_data.drop(othervars, axis=1)

#* Create squared variables 
ms_data['AGE_sq']=ms_data['AGE']**2
ms_data['chFAM_SIZE']=ms_data['chchildren']+ms_data['chadults']
ms_data['chFAM_SIZE_sq']=ms_data['chFAM_SIZE']**2

#* Dummies for Categoricals 
#categorical variables 
categoricals=['CKBK_CTX', 'MARITAL1', 'SAVA_CTX', 'CUTENURE']
#transform categoricals into dummies 
cats=ms_data[categoricals+['newid']]
cats_dummies=create_dummies(cats, 'newid')

#* Turn categoricals into two categories only 
#create dummies for categoricals that have several categories making a distinction between only two broader categories 
cats_single=cats.copy()
#married=1 if actually married, all other statuses=0
cats_single['married']=[int(x) for x in cats_single['MARITAL1']==1]
#owned_nm=1 if housing owned w/o mortgage 
cats_single['owned_nm']=[int(x) for x in cats_single['CUTENURE']==2]
#owned_m=1 if housing owned w/ mortgage 
cats_single['owned_m']=[int(x) for x in cats_single['CUTENURE']==1]
#notowned=1 if housing not owned (all other CUTENURE categories)
cats_single['notowned']=[int(x) for x in (cats_single['CUTENURE']!=1)&(cats_single['CUTENURE']!=2)]
#then drop the original vars 
cats_single=cats_single.drop(categoricals, axis=1)

#* Individual level dummies 
#create a df with individual level dummies 
ind_dums=ms_data[['custid', 'newid']]
ind_dums=create_dummies(ind_dums, 'custid')

#* Create quarter identifier 
#t dimension is quarter 
timedata=ms_data[meta]
#need a running count from first quarter in data to last one 
timedata['quarter']=1
for i in range(len(timedata)): 
    year=timedata.loc[i, 'QINTRVYR']
    month=timedata.loc[i, 'QINTRVMO']
    if year==2008:
        if month in [1, 2, 3]: 
            timedata.loc[i, 'quarter']=2
        elif month in [4, 5, 6]: 
            timedata.loc[i, 'quarter']=3
        elif month in [7, 8, 9]: 
            timedata.loc[i, 'quarter']=4
        elif month in [10, 11, 12]: 
            timedata.loc[i, 'quarter']=5
    elif year==2009: 
        timedata.loc[i, 'quarter']=6

#* Create dummies for samples 
#complete sample 
ms_data['comp_samp']=1
#sample with no NA 
ms_data['nona_samp']=ms_data['newid'].isin(ms_data.dropna()['newid']).astype(int)
#liquidity 
ms_data['l_samp']=gen_samp_dummy(ms_data, 'liqassii', 'newid')
#mortgage sample 
ms_data['mort_samp']=gen_samp_dummy(ms_data, 'ORGMRTX', 'newid')
#mortgage and liquidity 
ms_data['l_mort_samp']=(ms_data['mort_samp']==ms_data['l_samp']).astype(int)

#* Merge everything into one dataset
final=ms_data.merge(cats_single, on='newid')
#drop categoricals 
final=final.drop(categoricals, axis=1)
final=final.merge(cats_dummies, on='newid')
final=final.merge(timedata[['newid', 'quarter']], on='newid')

final.to_csv(data_out/'transformed'/'prepped_data.csv', index=False)