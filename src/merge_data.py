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
    #only keep variables
    sub_df=df[variables]
    #create id column 
    sub_df['id']=[int(var/10) for var in sub_df['NEWID']]
    #create interview column 
    sub_df['intrvw']=[int(str(var)[-1]) for var in sub_df['NEWID']]
    
    #return dataframe 
    return sub_df

#*####################
#! DATA 
#*####################
#first merge fmli data 
files=os.listdir(path=data_in/'2008_rawdata'/'intrvw08')
fmli=[f for f in files if 'fmli' in f]
#NEWID and variables that I want
variables=['EDUC0REF', 'NEWID']
#columns of final df to which all variables are appended
columns=['id', 'intrvw']+variables
final_df=pd.DataFrame(columns=columns)
#load each file, apply adjust_df() function to dataframe and append all into one
for f in fmli: 
    print(f)
    #read in file
    raw=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/f)
    #apply adjust_df()
    df_adjusted=adjust_df(raw, variables)
    #add a column signaling from which file the entry is taken
    df_adjusted['origin']=f.split('.')[0]
    #append to final_df
    final_df=final_df.append(df_adjusted)

#count how often each id appears
count_df=final_df
count_df['count']=0
count_df=final_df[['id', 'count']].groupby(['id']).count()
#filter for households that appear at least twice 
twice=count_df[count_df['count']>=2]
