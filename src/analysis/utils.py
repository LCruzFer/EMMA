import pandas as pd
import numpy as np 
import math
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import PolynomialFeatures
from econml.dml import LinearDML, CausalForestDML, SparseLinearDML
from econml.inference import BootstrapInference
import matplotlib.pyplot as plt 

'''
This file contains various helper functions for that are done in several files but always involve the same steps - e.g. splitting data into train and test sets or other things.
'''
class data_utils: 
    '''
    A class containing tools for data pre-processing and manipulation.
    '''
    def __init__(self): 
        pass

    def drop_vars(self, df, word): 
        '''
        Drop all columns of df that have 'word' in their name.
        '''
        df=df.drop([col for col in df.columns if word in col], axis=1)
        return df
    
    def get_expnames(self, df): 
        '''
        Filter self.df for all expenditure variables.
        *self.df=pandas dataframe, expenditure variables signaled by 'exp' in column name
        '''
        #get columns that have 'exp' in their name
        exp_cols=[col for col in df.columns if 'exp' in col]
        #also filter between level, change and lag 
        lvl=[col for col in exp_cols if ('ch' not in col) & ('last' not in col)]
        ch=[col for col in exp_cols if 'ch' in col]
        lag=[col for col in exp_cols if 'last' in col]
        return lvl, ch, lag

    def drop_exp_rbt(self, df):
        '''
        Drop all columns related to expenditures or rebate that are in self.df. 
        *self.df=pandas self.df
        '''
        df=df[[col for col in df.columns if 'exp' not in col]]
        df=df[[col for col in df.columns if 'rbt' not in col]]
        df=df[[col for col in df.columns if 'reb' not in col]]
        df=df[[col for col in df.columns if 'REB' not in col]]
        df=df[[col for col in df.columns if 'RBT' not in col]]
        return df

    def create_test_train(self, data, idcol, outcome, treatment, n_test=2000):
        '''
        Create single datasets for outcome, treatment and controls - train and test sets respectively. Households are either part of train or test set, not both.
        *data=df with all data 
        *idcol=column containing id of units 
        *outcome=str that is column name of outcome
        *treatment=str that is column name of treatment
        '''
        #set seed for replication 
        np.random.seed(1)
        #get IDs and draw n_test randomly from them 
        test_ids=np.random.choice(data[idcol], size=n_test, replace=False)
        #get observations for test & train set 
        test=data[data[idcol].isin(test_ids)]
        train=data[data[idcol].isin(test_ids)==False]
        #then split into different sets for outcome, treatment and confounders
        y_train=train[outcome]
        y_test=test[outcome]
        t_train=train[treatment]
        t_test=test[treatment]
        #remove all rebate or expenditure related variables from confounders data
        z_train=self.drop_exp_rbt(train[[col for col in train.columns if col not in [outcome, treatment]]])
        z_test=self.drop_exp_rbt(test[[col for col in train.columns if col not in [outcome, treatment]]])
        #then drop idcols from z data 
        z_train=z_train.drop(idcol, axis=1)
        z_test=z_test.drop(idcol, axis=1)
        
        return y_train, y_test, t_train, t_test, z_train, z_test

    def split_XW(self, Z, x_columns):
        '''
        Split all observables into X and W data.
        *Z=all observables 
        *x_columns=columns that are supposed to be in X
        '''
        #get x data
        x=Z[x_columns]
        #then get w data 
        w=Z[[col for col in Z.columns if col not in x.columns]]
        #if no columns in w, set it to None
        if w.shape[1]==0: 
            w=None
        return (x, w)

    def cross_fitting_folds(self, df, K, T, t_col): 
        '''
        Generate the cross-fitting folds from Chernozhukov et al. (2021) as a list of (train, test) indices. For K folds generate K pairs where each fold is test once and rest are train.
        *data=pandas dataframe containing all data 
        *K=number of folds 
        *T=number of periods 
        *t_col=column in data signaling period of observation
        '''
        #set up final list of folds 
        folds=[]
        #then for each fold 
        for k in range(1, K+1):
            #get upper and lower limit of time index 
            lower=np.floor(T*(k-1)/K+1)
            upper=np.floor(T*k/K)
            print(lower, upper)
            #then check which observations have lower<=t<=upper
            in_k_bools=(lower<=df[t_col])&(df[t_col]<=upper)
            #then get the corresponding indices 
            in_k_indices=df[in_k_bools].index
            #and oppposite of those 
            not_k_indices=df[in_k_bools==False].index
            #bind in tuple and add to fold list 
            folds.append((not_k_indices, in_k_indices))
        return folds

class fitDML(data_utils): 
    '''
    This class contains different functions to fit various DML estimators. 
    Class is initiated with supplying data, treatment and outcome and then splits them into train and test samples and makes necessary cleaning steps.
    Parent class is du from which we use some functions here.
    '''
    def __init__(self, data, treatment, outcome, x_cols, n_test=500): 
        '''
        data=pandas df 
        treatment=string with column name of treatment var 
        outcome=string with column name of outcome var 
        x_cols=list with columns names of X variables, rest are considered as W
        '''
        self.df=data
        self.treatment=treatment 
        self.outcome=outcome 
        self.x_cols=x_cols
        self.n_test=n_test
        #create test/train samples
        self.y_train, self.y_test, self.t_train, self.t_test, self.z_train, self.z_test=super().create_test_train(self.df, 'custid', self.outcome, self.treatment, n_test=self.n_test)
        #split z data into X and W
        self.x_train, self.w_train=super().split_XW(Z=self.z_train, x_columns=self.x_cols)
        self.x_test, self.w_test=super().split_XW(Z=self.z_test, x_columns=self.x_cols)
        #set up empty dict for CDFs
        self.cdfs={'linear': {}, 'cf': {}, 'sparse': {}}
        #set up empty dicts for ICE and PDP
        self.x_axis={}
        self.y_axis_pdp={'linear': {}, 'cf': {}, 'sparse': {}}
        self.y_axis_ice={'linear': {}, 'cf': {}, 'sparse': {}}

    def fit_linear(self, params_Y, params_T, folds): 
        '''
        Estimate a partially linear model using the DML approach. Estimation of E[Y|X] and E[T|X] is done using a random forest. 

        *params_Y=parameters for random forest for Y
        *params_T=parameters for random forest for T
        *folds=folds for CV, int or list 
        '''
        #initialize DML model with tuned parameters
        self.linDML=LinearDML(model_y=RFR(n_estimators=500,
                                    max_depth=params_Y['max_depth'],       
                                    min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), 
                        model_t=RFR(n_estimators=500,
                                    max_depth=params_T['max_depth'],    
                                    min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']), 
                        cv=folds, fit_cate_intercept=True
                        )
        print('Model set up!')
        #fit to train data 
        self.linDML.fit(self.y_train, self.t_train, X=self.x_train, W=self.w_train, 
                #bootstrapped or asymptotic inference?
                    #inference=BootstrapInference(n_bootstrap_samples=50)
                    )
        print('Model fitted!')
        self.lin_cate_df=self.linDML.const_marginal_effect_inference(X=self.x_test).summary_frame()
    
    def fit_cfDML(self, params_Y, params_T, folds): 
        '''
        Estimate a nonparametric model using the DML approach. Estimation of E[Y|X] and E[T|X] is done using a random forest. 

        *params_Y=parameters for random forest for Y 
        *params_T=parameters for random forest for T
        *folds=folds for CV, int or list 
        '''
        #initialize DML model with tuned parameters
        self.cfDML=CausalForestDML(model_y=RFR(n_estimators=500,
                                    max_depth=params_Y['max_depth'],       
                                    min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), 
                        model_t=RFR(n_estimators=500,
                                    max_depth=params_T['max_depth'],    
                                    min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']), 
                        cv=folds,
                        n_estimators=10000, 
                        drate=False, 
                        )
        print('Model set up!')
        #fit to train data 
        self.cfDML.fit(self.y_train, self.t_train, X=self.x_train, W=self.w_train, 
                #bootstrapped or asymptotic inference?
                    #inference=BootstrapInference(n_bootstrap_samples=50)
                    )
        print('Model fitted!')
        self.cf_cate_df=self.cfDML.const_marginal_effect_inference(X=self.x_test).summary_frame()

    def fit_sparseDML(self, params_Y, params_T, folds, feat): 
        '''
        Estimate a partially linear model using the DML approach. Estimation of E[Y|X] and E[T|X] is done using a random forest. 

        *params_Y=parameters for random forest for Y 
        *params_T=parameters for random forest for T
        *folds=folds for CV, int or list 
        *feat=featurizer, how to transform X
        '''
        #initialize DML model with tuned parameters
        self.spDML=SparseLinearDML(model_y=RFR(n_estimators=1000,
                                    max_depth=params_Y['max_depth'],       
                                    min_samples_split=params_Y['min_samples_leaf'], max_features=params_Y['max_features']), 
                        model_t=RFR(n_estimators=1000,
                                    max_depth=params_T['max_depth'],    
                                    min_samples_split=params_T['min_samples_leaf'], max_features=params_T['max_features']), 
                        featurizer=feat,
                        cv=folds, fit_cate_intercept=False, 
                    #! do they converge now?
                        max_iter=2000
                        )
        print('Model set up!')
        #fit to train data 
        self.spDML.fit(self.y_train, self.t_train, X=self.x_train, W=self.w_train, 
                #bootstrapped or asymptotic inference?
                    #inference=BootstrapInference(n_bootstrap_samples=50)
                    )
        print('Model fitted!')
        self.sp_cate_df=self.spDML.const_marginal_effect_inference(X=self.x_test).summary_frame()
    
    def selectmodel(self, model): 
        if model=='linear': 
            return self.linDML
        elif model=='cf':
            return self.cfDML
        elif model=='sp':
            return self.spDML
        else: 
            raise ValueError('Model must be linear, cf or sp')

    def meam(self, var, model='linear'): 
        '''
        Calculate marginal effect at mean (meam) for var. 
        
        *var=str, column name of variable in self.df
        *model=str, choose for which model meam should be calculated
        '''
        #first select correct model 
        estim=self.selectmodel(model=model)
        #get means of X data 
        means=self.x_test.mean()
        means_df=self.x_test.copy() 
        for col in self.x_test.drop(var, axis=1).columns:
            means_df[col]=means[col]
        #then calculate MEAM and get inference df
        meam=estim.const_marginal_effect_inference(X=means_df).summary_frame()
        #then get summary frame
        meam_df=meam.summary_frame() 
        
        return meam_df
    
    def get_cdfs(self, model):
        '''
        Get empirical cdf of point estimate and CI bounds for all estimations of specification.
        '''
        #select model
        if model=='linear':
            cate_df=self.lin_cate_df
        elif model=='cf':
            cate_df=self.cf_cate_df
        elif model=='sp':
            cate_df=self.sp_cate_df
        else: 
            raise ValueError('Model must be linear, cf or sp')
        lines=['point_estimate', 'ci_lower', 'ci_upper']
        for line in lines: 
            #point estimates sorted by size 
            pe_sorted=np.sort(np.array(cate_df[line]))
            #proportional values of sample 
            proportions=np.arange(len(pe_sorted))/(len(pe_sorted)-1)
            #and save as attribute of class
            self.cdfs[model][line]=(pe_sorted, proportions)

    def pdp(self, var, model='lin', alpha=0.1): 
        '''
        Create partial dependence plot x- and y-axis values and save them in dictionary as attribute of class. 
        *var=str, column name of variable in self.df
        *model=str, choose for which model mean should be calculated
        *alpha=float, CI bounds will be calculated for alpha and 1-alpha
        '''
        #first select correct model 
        estim=self.selectmodel(model=model)
        #select data 
        data=self.x_test
        #get min and max values
        low=min(data[var])
        high=max(data[var])
        #create range 
        #if binary: range just low and high
        if (low==0)&(high==1): 
            x_axis=np.array((0, 1))
        else:
            #if not binary, make x axis contain each percentile value  
            pctiles=[int(data[var].quantile(i)) for i in np.round(np.linspace(0, 1, 100), decimals=2)]
            x_axis=np.array(pctiles)
#            x_axis=np.linspace(low, high, min(100, int(high-low))).astype(int)
        #set up empty array in which y axis values for pdp will be saved
        y_axis=np.empty((x_axis.shape[0], 5))
        #fit model for each value 
        for i, val in enumerate(x_axis):
            #create copy of data
            copy=data.copy() 
            #set variable to value
            copy[var]=val
            #then get ATE, CIs and stderr at this point
            ate_inf=estim.const_marginal_ate_inference(X=copy)
            ate_ci=ate_inf.conf_int_mean(alpha=alpha)
            y_axis[i, :]=np.array((ate_ci[0], ate_inf.mean_point, ate_ci[1], ate_inf.stderr_mean, ate_inf.pvalue()))
        #save as attributes of the class
        self.x_axis[var]=x_axis
        self.y_axis_pdp[model][var]=y_axis
    
    def all_pdp_axis(self, model='linear', alpha=0.1): 
        '''
        Get all pdp axis for vars using specified model.
        '''
        #then for each X variable apply pdp_function 
        variables=self.x_cols
        for var in variables: 
            self.pdp(var, model=model, alpha=alpha)
            print(var+' done')
    
    def ice(self, var, model='lin'):
        '''
        Calculate individual conditional expectation values for var. 
        *var=str, column name of variable in self.df
        *model=str, choose for which model mean should be calculated
        '''
        #first select correct model 
        estim=self.selectmodel(model=model)
        #select data 
        x_data=self.x_test
        t_data=self.t_test
        #get min and max values
        low=min(x_data[var])
        high=max(x_data[var])
        #create range 
        #if binary: range just low and high
        if (low==0)&(high==1): 
            x_axis=np.array((0, 1))
        else: 
            #if not binary, make x axis contain each percentile value  
            #use nearest value as need values that exist in data to find index
            pctiles=[int(x_data[var].quantile(i, interpolation='nearest')) for i in np.round(np.linspace(0, 1, 50), decimals=2)]
            # #need indices to filter respective treatment data
            #!don't need those necessarily, I am a bit stupid here
            # pct_ids=[np.where(x_data[var]==pct) for pct in pctiles]
            #get x_axis
            x_axis=np.array(pctiles)
            #x_axis=np.linspace(low, high, min(100, int(high-low))).astype(int)
            #get relevant treatment data 
            #if multiple observations are at pctile, use them all 
            # t_pctiles=np.array([np.mean(t_data.iloc[i]) for i in pct_ids])
        y_axis=np.empty((x_axis.shape[0], len(x_data[var]), 5))
        for i, val in enumerate(x_axis):
            print(i)
            #create copy of data
            x_copy=x_data.copy() 
            #set variable to value
            x_copy[var]=val
            #then get ATE, CIs and stderr at this point
            cate_df=estim.const_marginal_effect_inference(X=x_copy).summary_frame()
            y_axis[i, :, :]=np.array((cate_df['ci_lower'], cate_df['point_estimate'], cate_df['ci_upper'], cate_df['stderr'], cate_df['pvalue'])).T
        self.x_axis[var]=x_axis
        self.y_axis_ice[model][var]=y_axis

    def all_ice_axis(self, vars, model): 
        '''
        Get ICE for all variables in vars.
        '''
        for var in vars: 
            self.ice(var, model=model)