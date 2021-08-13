from pathlib import Path 
import pandas as pd 

#* set path 
wd=Path.cwd()
data_in=wd.parent/'data'/'in'
data_out=wd.parent/'data'/'out'

'''
This file tries to extend the Parker et al. (2013) dataset (also used by Misra & Surico) with additional variables that are not included but might be of interest - e.g. education of reference person. 
'''

#*####################
#! DATA 
#*####################
#load Parker et al. data 
parker=pd.read_stata(data_in/'Parkeretal_Data.dta')
