import sys 
from pathlib import Path 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils as du 
from utils import estim_utils as eu
#* set data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'

'''
Idea: since the shock I investigate is anticipated, households that react less when shock realizes should show some reaction in their savings - in the sense that they decline - because they smooth consumption while those reacting strongly should be at borrowing constraint and not be able to smooth consumption.
The tax stimulus program was enacted in February 2008.
I have data on balance of checkings and savings account as well as mortgages.
'''

#*#########################
#! FUNCTIONS
#*#########################

#*#########################
#! DATA
#*#########################
