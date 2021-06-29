from pathlib import Path 
import os
import pandas as pd 

#* set path 
wd=Path.cwd()
data_in=wd.parent/'data'/'in'
data_out=wd.parent/'data'/'out'

'''
This file merges the single quarter datasets downloaded from the CEX website into one and filters them for the characteristics of interest.
'''
#*####################
#! FUNCTIONS
#*####################
def adjust_df(df, variables): 
    '''
    Adjust dataframe 'df' to only keep 'variables' and create an interview column and an id column from 'NEWID'.
    
    *df=pandas dataframe, needs to have a NEWID columns
    *variables=list of variables that should be kept
    '''
    #make sure that df contains NEWID column
    if 'NEWID' not in df.columns: 
        raise KeyError('NEWID not in columns!')
    #append NEWID to variables
    variables.append('NEWID')
    #only keep variables
    df=df[variables]
    #create id column 
    df['id']=[int(var/10) for var in df['NEWID']]
    #create interview column 
    df['intrvw']=[int(str(var)[-1]) for var in df['NEWID']]
    
    #return dataframe 
    return df

#*####################
#! DATA 
#*####################
#first merge fmli data 
files=os.listdir(path=data_in/'2008_rawdata'/'intrvw08')
fmli=[f for f in files if 'fmli' in f]
variables='EDUC0REF'
columns=list(pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli082.csv').columns)+['id', 'intrvw']
final_df=pd.DataFrame(columns=columns)
for f in files: 
    print(f)
    df=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/f)
#    df['NEWID']=df['NEWID'].astype(int)
    df_adjusted=adjust_df(df, variables)
    final_df=df_adjusted.append(df_adjusted)