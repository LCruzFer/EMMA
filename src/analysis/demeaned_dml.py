from pathlib import Path
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, train_test_split
import utils

#* set paths 
wd=Path.cwd()
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

#*#########################
#! FUNCTIONS
#*#########################

#*#########################
#! DATA
#*#########################
#read in demeaned data
demeaned_vars=pd.read_csv(data_out/'transformed'/'prepped_data_demeaned.csv')
#choose outcome 
outcome='RBTAMT'
#choose treatment
treatment='RBTAMT'
#split into test and train data and then create subset dataframes
y_train, y_test, r_train, r_test, z_train, z_test=utils.create_test_train(demeaned_vars, outcome, treatment)
#drop all other expenditure and rebate variables from Z data 
z_train=utils.drop_exp_rbt(z_train)
z_test=utils.drop_exp_rbt(z_test)

#read in hyperparameters for RF - output from tune_first_stage.py

#*#########################
#! Linear DML
#*#########################
'''
Here consider DML model with semi-parametric specification: 
Y_{it}=\theta*R_{it} + g(X_{it}, W_{it}) + U_{it}
R_{it}=f(X_{it}, W_{it}) + V_{it}
For estimation of g() and f() a regression forest is used (for now - explore other options later on) 
The hyperparameters for the forest are tuned in tune_first_stage.py and contained in 'hyperparams_rf' dataframe
'''
