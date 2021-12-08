import sys 
import os
from pathlib import Path
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#* set data paths
wd=Path.cwd()
sys.path.append(str(wd.parent))
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'
fig_out=wd.parents[1]/'figures'
results=wd.parents[1]/'data'/'results'

#*#########################
#! FUNCTIONS
#*#########################
def get_ratio(df, alpha):
    '''
    Find what share of hh has significant response at confidence level alpha.
    '''
    ratio=sum(df['pvalue']<=alpha)/len(df)
    return ratio

def ratios_variable(variable, alpha):
    '''
    Get ratios of significance for outcome=variable across all specifications and estimators.
    '''
    #get list of result dfs
    files=os.listdir(results/variable)
    files.sort()
    #save them in dictionary that is then turned into pandas df
    ratio_dict={}
    for file in files: 
        df=pd.read_csv(results/variable/file)
        ratio=get_ratio(df, alpha=alpha)
        #ugly but gets a nice way of naming for dictionary
        dict_name=file.split('_')[1]+'_'+file.split('_')[2].split('.')[0]
        ratio_dict[dict_name]=ratio
    ratio_df=pd.DataFrame.from_dict(ratio_dict, orient='index')
    return ratio_df

def get_all_ratios(variables, alpha):
    '''
    Get ratio df for all variables in variables list.
    '''
    indices=['spec1_cf', 'spec1_lin', 'spec2_cf', 'spec2_lin', 'spec3_cf', 'spec3_lin', 'spec4_cf', 'spec4_lin']
    final_df=pd.DataFrame(index=indices)
    for var in variables:
        df_var=ratios_variable(var, alpha)
        df_var=df_var.rename(columns={0: var})
        #merge onto final_df
        final_df=final_df.merge(df_var, left_index=True, right_index=True)
    return final_df

def str_to_tex(file, tex_str):
    tex_file=open(results/'significance_ratios'/file, mode='a')
    tex_file.write(tex_str)
    tex_file.close()

#*#########################
#! DATA
#*#########################
variables=['chTOTexp', 'chNDexp', 'chSNDexp', 'chFDexp',]
#get ratios for alpha=0.1
ratios_10pct=get_all_ratios(variables, alpha=0.1)
filename='ratios_10pct.tex'
str_to_tex(filename, ratios_10pct.to_latex())
ratios_5pct=get_all_ratios(variables, alpha=0.05)
filename='ratios_5pct.tex'
str_to_tex(filename, ratios_5pct.to_latex())
