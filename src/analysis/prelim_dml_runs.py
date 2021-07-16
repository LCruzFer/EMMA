from pathlib import Path
import numpy as np
import pandas as pd 
import econml
from sklearn.ensemble import RandomForestRegressor

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

#*#########################
#! DATA
#*#########################
#for preliminary analysis, use the Parker et al. and Misra and Surico datasets
ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/
                    '2008_data.csv')
parker_data=pd.read_stata(data_in/'Parkeretal_Data.dta')

#for now take a look at MS data as they have more variables 
#use all observable characteristics as X 
observables=['chadults', 'chchildren', 'AGE', 'adults', 'children', 'CUTENURE', 'FINCBTXM', 'FSALARYM', 'MARITAL1', 'RESPSTAT', 'SAVA_CTX']
cols=list(ms_data.columns)
ms_data['SAVA_CTX']
#create a dummy category that are only used in final estimation 
dummies=['timetrend' 'yymm']
#household identifiers
#custid is newid with interview no (last digit) removed - unique household identifier
ids=['newid', 'interviewno', 'custid']
#variables related to rebate - see main-columns.csv of MS for explainers
rebate_vars=['taxreb', 'itaxreb', 'ltaxreb', 'iltaxreb', 'itotalreb']
#now create dataframes with subsets 
X=ms_data[observables]
#some observables are categoricals, which I hence transform into integer codes -> ignore those for now as they make problems in sklearn's RFR class 
#need to do OneHotEncoding, which is computationally expensive, see https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree

dummies=ms_data[dummies]
R=ms_data[rebate_vars]
IDs=ms_data[ids]

#*#########################
#! ANALYSIS
#*#########################
#start with predicting rebate amount 'taxreb' using the observables in X
