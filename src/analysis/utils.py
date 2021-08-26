import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 

'''
This file contains various helper functions for that are done in several files but always involve the same steps - e.g. splitting data into train and test sets or other things.
'''
class data_utils: 
    '''
    A class containing tools for data pre-processing and manipulation.
    '''
    def __init__(self): 
        pass

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
        return (x, w)

class estim_utils: 
    '''
    Class providing tools for estimation steps that are done across different estimation specifications several times.
    '''
    def __init__(self): 
        pass 
    
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

    def marginal_effect_at_means(self, model, x_test, var):
        '''
        Provided a fitted model and test data, this function generates a dataset that has all columns at their means except var column and then calculates the constant marginal effect at the means of this var.
        *model=fitted DML model
        *x_test=test data 
        *var=variable that MEAM should be calculated for 
        '''
        #get means of X data 
        means=x_test.mean()
        means_df=x_test.copy() 
        for col in x_test.drop(var, axis=1).columns:
            means_df[col]=means[col]
        #then calculate MEAM and get inference df
        meam=model.const_marginal_effect_inference(X=means_df)
        #then get summary frame 
        meam_df=meam.summary_frame() 
        
        return meam_df

    def get_all_meam(self, model, x_test): 
        '''
        Apply marginal_effect_at_means() to all variables in X test set and combine results in dataframe.
        *model=fitted econML DML model that can calculate constant_marginal_effects
        *x_test=test set of X variables
        '''
        #get columns of test data 
        x_cols=list(x_test.columns)
        #calculate MEAM for first variable in X data to create a df on which rest is later merged 
        meams=self.marginal_effect_at_means(model, x_test, x_cols[0])
        meams=meams.rename(columns={col: col+'_'+x_cols[0] for col in meams.columns})
        #then loop over all variables in x_test and merge the results to meams df 
        for var in x_cols[1:]: 
            meam_var=self.marginal_effect_at_means(model, x_test, var)
            #only keep point estimate, SE and pvalue
            meam_var=meam_var.rename(columns={col: col+'_'+var for col in meam_var.columns})
            meams=meams.merge(meam_var, left_index=True, right_index=True)
        
        return meams
    
    def coef_var_correlation(self, df, coefs): 
        '''
        Calculate Pearson's coefficients of correlation between observables and individual treatment effects. 
        
        *df=pd dataframe with X data 
        *coefs=pd series containint treatment effect coefficients
        '''
        #save correlations in dict
        corrs={}
        #for each column in X data calculate correlation with coefficients
        for col in df.columns: 
            corrs[col]=np.corrcoef(df[col], coefs)[0, 1]
        
        return corrs