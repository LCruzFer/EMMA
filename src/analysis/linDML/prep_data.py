from pathlib import Path
import pandas as pd

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

'''
This file prepares 'raw' data and creates CSVs containing the prepped data - one with demeand variables, one without.
Steps taken: 
- turn categoricals into dummies via OneHotEncoding
- demean all observations by their time mean on individual level 
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

def demean_df(df, idcol, ignorecol=None):
    '''
    Demean all observations in df using individual level mean over time. 
    *df=dataframe containing variables to be demeaned
    *idcol=str; signaling id column of df; level of mean
    *ignorecol=str; ignore this column in demeaning, e.g. variable signaling period
    '''
    #get mean on idcol level 
    means=df.groupby(idcol).mean()
    #rename columns to column_mean
    means=means.rename(columns={col:col+'_mean' for col in means.columns})
    #make m:1 merge back onto original df 
    all_vars=df.merge(means, on=idcol)
    #then create new df with demeaned variables 
    demeaned_vars=all_vars.copy()
    #df contains original variables we want to demean
    #loop over these columns
    #ignore the idcol and ignorecol
    collist=list(df.columns)
    collist.remove(idcol)
    collist.remove(ignorecol)
    for col in collist: 
        #create demeaned version of variable
        demeaned_vars[col]=all_vars[col]-all_vars[col+'_mean']
        #and drop not demeaned variable and mean
        demeaned_vars=demeaned_vars.drop([col+'_mean'], axis=1)
    return demeaned_vars

#*#########################
#! DATA
#*#########################
ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/'2008_data.csv')
#change order of columns in ms_data to display relevant IDs as first columns 
ms_data=reorder_df(['custid', 'interviewno', 'newid'], ms_data)

#*Categoricals 
#some observables are categoricals, which need to be encoded using OneHotEncoding
#-> computationally expensive, see https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree
#-> how to do it: https://stackoverflow.com/questions/24706677/how-to-handle-categorical-variables-in-sklearn-gradientboostingclassifier/24874515#24874515
#first create a list with categoricals
categoricals=['CKBK_CTX', 'MARITAL1', 'SAVA_CTX', 'ST_HOUS']
#create a df with categoricals and newid
cats=ms_data[categoricals+['newid']]
#OneHotEncoding 
cats_dummies=create_dummies(cats, 'newid')
#drop the raw variables from the original df and merge dummy vars back onto it 
variables=ms_data.drop(categoricals, axis=1)
variables=variables.merge(cats_dummies, on='newid')

variables.to_csv(data_out/'transformed'/'nondemeaned_vars.csv', index=False)

#*Demeaning Individual Level 
#for demeaning can ignore all variables that are constants signaling month (const1-const15) as well as timetrend column 
demean_vars=variables.drop(['const', 'newid', 'interviewno']+['const'+str(i) for i in range(15)], axis=1)
#!UNNECESSARY as dependent variable is first difference (change in consumption)
#demean based on individual level 
#i_demeaned_vars=demean_df(demean_vars, 'custid', 'timetrend')
#*Demeaning Time Level
#now demean based on timetrend 
it_demeaned_vars=demean_df(demean_vars, 'timetrend', 'custid')
#remove timetrend and custid mean columns 
it_demeaned_vars=it_demeaned_vars.drop(['custid_mean'], axis=1)

#*Subset with complete data only 
#some variables contain lots of NaN values in the above derived dataframe 
#due to NaN values in the original data 
#find out whether this is due to certain custid having NaN only over all periods or whether this changes
nan_cols=[]
for col in it_demeaned_vars.columns: 
    nan_series=it_demeaned_vars[col].isnull()
    if sum(nan_series)!=0: 
        nan_cols.append(col)
for cu in it_demeaned_vars['custid'].drop_duplicates(): 
    cu_df=it_demeaned_vars[it_demeaned_vars['custid']==cu]
#pattern clearly shows that his is changing over time 
sum(it_demeaned_vars['QESCROWX'].isnull())
#hence, drop all rows in which there is NaN 
no_missings=it_demeaned_vars.dropna()

#*CSVs 
#don't write dedicated index column to CSV
it_demeaned_vars.to_csv(data_out/'transformed'/'prepped_data_demeaned.csv', index=False)
no_missings.to_csv(data_out/'transformed'/'demeaned_vars_nomissings.csv', index=False)