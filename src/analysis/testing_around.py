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
data_in=wd.parents[1]/'data'/'in'
data_out=wd.parents[1]/'data'/'out'

ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/'2008_data.csv')
parker=pd.read_stata(data_in/'Parkeretal_Data.dta')
parker['custid']=parker['newid']
parker['newid']=(parker['newid'].astype(str)+parker['intview'].astype(str)).astype(int)
list(parker.columns)
ms_data=ms_data.merge(parker[['newid', 'liqassii']], left_on='newid', right_on='newid')
#generate AGE^2 
ms_data['age_sq']=ms_data['AGE']**2
ms_data['chFAM_SIZE']=ms_data['chchildren']+ms_data['chadults']
ms_data['chFAM_SIZE_sq']=ms_data['chFAM_SIZE']**2

#subset for ms base estimation: only age, age_sq, change in fam_size and change in fam_size squared and rebate and consumption variables
ms_subset=ms_data[['custid', 'newid', 'QINTRVYR', 'QINTRVMO', 'RBTAMT', 'AGE', 'age_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']]
#no NAN
#how many obs per period 
ms_counted=ms_subset.groupby(['QINTRVYR', 'QINTRVMO']).count()

#my own subset of interest 
list(ms_data.columns)
vars_of_interest=['custid', 'newid', 'QINTRVYR', 'QINTRVMO', 'RBTAMT', 'AGE', 'SAVA_CTX', 'MARITAL1', 'CKBK_CTX', 'QESCROWX', 'QMRTTERM', 'ORGMRTX', 'FINCBTXM', 'FSALARYM', 'liqassii']
othervars=[col for col in ms_data.columns if col not in vars_of_interest]
own=ms_data[vars_of_interest]
for var in vars_of_interest: 
    copy=own.copy() 
    nona=copy[copy[var].isnull()==False]
    len_nona=len(nona)
    print(f'{var}: {len_nona} observations')

for var in othervars: 
    copy=ms_data.copy() 
    nona=copy[copy[var].isnull()==False]
    len_nona=len(nona)
    print(f'{var}: {len_nona} observations')

#take a look at raw data what NaN actually is 
mor07=pd.read_csv(data_in/'2007_rawdata'/'expn07'/'mor07.csv')
mor07['custid']=mor07['NEWID'].apply(lambda x: int(str(x)[:-1]))
mor07_ids=mor07[['NEWID', 'custid']]
mor07_ids['interviewno']=mor07_ids['NEWID'].apply(lambda x: int(str(x)[-1]))
mor07_ids=mor07_ids.drop_duplicates(['NEWID'])
mor07_idsg=mor07_ids.groupby(['custid']).count()
mor07_ids=mor07_ids.merge(mor07_idsg, on='custid')
mor07_ids[mor07_ids['NEWID_y']!=1]
mor07[mor07['custid']==175049]
mor08=pd.read_csv(data_in/'2008_rawdata'/'expn08'/'mor08.csv')
mor08['custid']=mor08['NEWID'].apply(lambda x: int(str(x)[:-1]))
mor08_ids=mor08[['NEWID', 'custid']].drop_duplicates(['custid'])

ms_ids07=ms_data[['newid', 'custid']][ms_data['QINTRVYR']==2007].drop_duplicates(['custid'])

overlap=ms_ids07.merge(mor08_ids, on='custid', how='left', indicator=True)
sum(overlap['_merge']=='both')

fmli074=pd.read_csv(data_in/'2007_rawdata'/'intrvw07'/'fmli074.csv')
fmli074[fmli074['QINTRVMO']==12]['NEWID']

list(fmli074.columns)
var='SAVA_CTX'
copy=fmli074.copy() 
nona=copy[copy[var].isnull()==False]
len_nona=len(nona)
print(f'{var}: {len_nona} observations')
sum(fmli074['CKBKACTX'].isnull()==False)

ms_data[ms_data['QINTRVYR']==2007]['interviewno']
sum(fmli074['SAVACCTX'].isnull()==False)

ms_07ids=ms_data[['custid', 'interviewno']][ms_data['QINTRVYR']==2007].drop_duplicates('custid')
ms_08ids=ms_data[['custid', 'interviewno']][ms_data['QINTRVYR']==2008].drop_duplicates('custid')

overlap=ms_07ids.merge(ms_08ids, on='custid', indicator=True, how='left')
len(overlap[overlap['_merge']!='both'])

sum(parker['liqassii'].isnull())
#* how large is variation for observed characteristics within households? 
sum(ms_data[['custid', 'FSALARYM']].groupby(['custid']).apply(np.var)['FSALARYM']!=0)

countid=ms_data[['custid', 'FSALARYM']].groupby(['custid']).count()
morethanonce=countid[countid['FSALARYM']==3].index

cols=list(parker.columns)
cols.sort()
countid=final_smallest[['custid', 'FSALARYM']].groupby(['custid']).count()
morethanonce=countid[countid['FSALARYM']==3].index
morethanonce_data=parker[parker['custid'].isin(morethanonce)]
sum(morethanonce_data[['custid', 'FSALARYM']].groupby(['custid']).apply(np.var)['income']!=0)

single_hh_test=final_smallest[final_smallest['custid']==183913][x_cols].dropna(axis=1)
final_smallest[final_smallest['custid']==192728]['RBTAMT']
spec1_inf=spec1.const_marginal_effect_inference(X=single_hh_test).summary_frame()

x_testsameage=x_test[x_test['AGE']==50]
sameage_id=list(x_test[x_test['AGE']==50].index)
r_test.iloc[sameage_id, ]
spec3_inf=spec3.const_marginal_effect_inference(X=x_testsameage).summary_frame()

#* by quarter checking where rbtamt!=0
ms_quarter=ms_data[['newid', 'custid']+treatment].merge(timedata[['newid', 'quarter']], on='newid')

sum(ms_quarter.groupby(['custid']).sum()['iREB']!=0)

sum(ms_quarter['RBTAMT'][ms_quarter['RBTAMT']>0]==1200)
(935+796)/3027
ms_quarter['RBTAMT'].sum()

ms_quarter[ms_quarter['quarter']==3]


sum(parker['hhinrbtfile']!=1)
(parker['esp']).mean()

(ms_data['RBTAMT_CHK']+ms_data['RBTAMT_EF']).sum()
ms_data['RBTAMT'].sum()

parkernew=pd.read_stata('/Users/llccf/Downloads/116117-V1/ParkerSoulelesJohnsonmMcClellandReplication/PSJMReplication.dta')
parkernew['esp']


rawrbt=pd.read_csv(data_in/'2008_rawdata'/'expn08'/'rbt08.csv')
rawrbt=rawrbt[['NEWID', 'CUID', 'RBTAMT']].rename(columns={'RBTAMT': 'rawRBT'})

raw_and_ms=ms_data[['newid', 'custid', 'RBTAMT']].merge(rawrbt, left_on='newid', right_on='NEWID', how='outer', indicator=True)
raw_and_ms[raw_and_ms['_merge']=='right_only']['NEWID']

sum(rawrbt.groupby(['CUID']).sum()['RBTAMT']!=0)
rawrbt['RBTAMT'].sum()
rawrbt09=pd.read_csv(data_in/'2009_rawdata'/'intrvw09'/'stimulus09_addendum'/'stimulus_add_093.csv')
rawrbt09['QSTIMPYX'][rawrbt09['QSTIMPYX']=='.']=np.nan
rawrbt09['QSTIMPYX']=rawrbt09['QSTIMPYX'].astype(float)
rawrbt09['QSTIMPYX'].sum()

grouped=parker.groupby(['custid']).sum()
sum(grouped['esp']!=0)
mergeparker=parker[['custid', 'esp']].merge(grouped[['esp']], left_on='custid', right_on='custid', how='left')

mergeparker[mergeparker['esp_x']!=mergeparker['esp_y']]
mergeparker[mergeparker['custid']==194968]

ms_data.groupby(['custid']).sum()['RBTAMT']==0
5119/len(ms_data.groupby(['custid']).sum())

#! WEIGHTS STUFF 
fmli082=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli082.csv')
fmli082['TOTexp']=0
for var in expvars_current: 
    fmli082['TOTexp']=fmli082['TOTexp']+fmli082[var]
fmli082['lastTOTexp']=0
for var in expvars_past: 
    fmli082['lastTOTexp']=fmli082['lastTOTexp']+fmli082[var]

ms_data=pd.read_csv(data_in/'Misra_Surico_Data'/'2008_data.csv')
ms_data08=ms_data[ms_data['QINTRVYR']==2008][['newid', 'custid', 'chTOTexp', 'lastTOTexp', 'TOTexp', 'QINTRVMO']]
ms_data082=ms_data08[ms_data08['QINTRVMO'].isin([4, 5, 6])]

relevant=ms_data082.merge(fmli082[['NEWID', 'TOTexp', 'lastTOTexp']], left_on='newid', right_on='NEWID', how='left')

relevant[['newid', 'TOTexp_x', 'TOTexp_y']]

fmli082_sas=pd.read_sas(data_in/'2008_sas'/'intrvw08'/'fmli082.sas7bdat')
fmli082_sas['TOTexp']=0
for var in expvars_current: 
    fmli082_sas['TOTexp']=fmli082_sas['TOTexp']+fmli082_sas[var]

parker=pd.read_stata(data_in/'parkeretal_Data.dta')
parker['custid']=parker['newid']
parker['newid']=(parker['newid'].astype(str)+parker['intview'].astype(str)).astype(int)
parker[['newid', 'weight']].merge(fmli082[['NEWID', 'FINLWT21']], left_on='newid', right_on='NEWID')

sum(ms_data.groupby(['custid']).sum()['iREB']==1)


totw=parker['weight'].sum()
sum(parker['esp']*parker['weight'])
ms_data=ms_data.merge(parker[['newid', 'weight']], on='newid', how='left')
test=ms_data[['newid', 'weight', 'TOTexp', 'QINTRVYR', 'QINTRVMO']].merge(fmli074[['NEWID', 'FINLWT21', 'TOTexp']], left_on='newid', right_on='NEWID')

test['share']=test['TOTexp_x']/test['TOTexp_y']
test['w_share']=test['weight']

'''
weighting example from R example: 
popwt = ifelse(
            qintrvmo %in% 1:3 & qintrvyr %in% year,
            (qintrvmo - 1) / 3 * finlwt21 / 4,
            ifelse(
                qintrvyr %in% (year + 1),
                (4 - qintrvmo) / 3 *finlwt21 / 4,
                finlwt21 / 4
            )
        )
'''
#!but what is done with this?
ms_data['popw']=0
year=2008
for i in range(ms_data.shape[0]):
    mo=ms_data.loc[i, 'QINTRVMO']
    yr=ms_data.loc[i, 'QINTRVYR']
    wt=ms_data.loc[i, 'weight']
    if (mo in [1, 2, 3])&(yr==year):
        ms_data.loc[i, 'popw']=(mo-1)/3*wt/4 
    else:
        if yr==year+1:
            ms_data.loc[i, 'popw']=(4-mo)/3*wt/4
        else: 
            ms_data.loc[i, 'popw']=wt/4

parker['']
parker['esp'][parker['esp']>0].mean()

import statsmodels.api as sm

parker['month']=parker['yymm'].apply(lambda x: float(str(x)[1:]))
parker['year']=parker['yymm'].apply(lambda x: 2000+int(str(x)[0]))
parker=parker.merge(pd.get_dummies(parker['month']).rename(columns={i: 'const'+str(i) for i in range(1, 13)}), left_index=True, right_index=True)
list(parker.columns)
X=parker[['esp', 'age', 'dnad', 'dnkd']+['const'+str(i) for i in range(1, 13)]]
X=sm.add_constant(X)
y=parker['dcs']
reg=sm.OLS(y, X)
res=reg.fit(cov_type='cluster', cov_kwds={'groups': parker['custid']}) 
print(res.summary())

ms_data['AGE_sq']=ms_data['AGE']**2
ms_data['chFAM_SIZE']=ms_data['chchildren']+ms_data['chadults']
ms_data['chFAM_SIZE_sq']=ms_data['chFAM_SIZE']**2

X=ms_data[['RBTAMT', 'AGE', 'AGE_sq', 'chFAM_SIZE', 'chFAM_SIZE_sq']+['const'+str(i) for i in range(0, 15)]]
y=ms_data['chFDexp']
def fit_qr(q):
    qreg=sm.QuantReg(y, X)
    qres=qreg.fit(q=q)
    return(qres.params['RBTAMT'])
qs=np.arange(0.05, 1, 0.05)
results=[fit_qr(q) for q in qs]

import matplotlib.pyplot as plt 

plt.plot(qs, results)



main_mo=[804, 805, 806, 807]

receivedrbt=parker[['custid', 'iesp']].groupby('custid').sum()
receivedrbt['received']=(receivedrbt['iesp']>=1)

subset=parker[['custid', 'newid', 'yymm']].merge(receivedrbt, on='custid', how='left')

rcvd_main=len(subset[(subset['received']==True)&(subset['yymm'].isin(main_mo))])
rcvd_all=sum(receivedrbt['received']==True)

rcvd_main/rcvd_all

hh_received=parker[parker['esp']>0]
hh_rcvd_main=hh_received[hh_received['yymm'].isin(main_mo)]

rbt08=pd.read_csv(data_in/'2008_rawdata'/'expn08'/'rbt08.csv')
cex085=rbt08[rbt08['RBTMO']==5]
fmli082=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli082.csv')
fmli085=fmli082[fmli082['QINTRVMO']==5]
rbt085=cex085.merge(fmli085[['NEWID', 'FINLWT21']], left_on='NEWID', right_on='NEWID')

rbt085['popw']=(4-1)/3*rbt085['FINLWT21']/4 

sum(rbt085['RBTAMT']*rbt085['popw'])

parker['iesp']==1
parker['esp']>0
list(parker.columns)
sum(parker['sample3'])

parker[['newid', 'custid', 'iesp', 'sample3']]

fmli091=pd.read_csv(data_in/'2009_rawdata'/'intrvw09'/'fmli091.csv')
fmli091['custid']=fmli091['NEWID'].apply(lambda x: int(str(x)[:-1]))
rbt08=pd.read_csv(data_in/'2008_rawdata'/'expn08'/'rbt08.csv')
rbt08['custid']=rbt08['NEWID'].apply(lambda x: int(str(x)[:-1]))
rbt08[rbt08['custid']==197325]
parker[parker['custid']==197325]
fmli084=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli084.csv')
fmli084['custid']=fmli082['NEWID'].apply(lambda x: int(str(x)[:-1]))
fmli084[fmli084['custid']==197325]
parker[parker['newid']==1973253]
fmli091x=pd.read_csv(data_in/'2009_rawdata'/'intrvw09'/'fmli091x.csv')
fmli091x[fmli091x['NEWID']==1973253]['QINTRVMO']

