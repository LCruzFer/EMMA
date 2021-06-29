from pathlib import Path
import pandas as pd 

#* set path 
wd=Path.cwd()
data_in=wd.parent/'data'/'in'
data_out=wd.parent/'data'/'out'

'''
This file is used for first exploration of the Public Use Microdata (PUMD) from the Consumer Expenditure Survey (CEX). 
Steps done: 
1. explore overlap of household IDs in different quarters
2. filter list of variables that are in PUMD dictionary for relevant period
'''

#*####################
#! DATA EXPLORATION
#*####################

#* STEP I: explore NEWID overlap
#let's look at the fmli files from second and third quarter in 2008
fmli082=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli082.csv')
fmli083=pd.read_csv(data_in/'2008_rawdata'/'intrvw08'/'fmli083.csv')
#get NEWID column for respective quarter and remove quarter identifier (last digit)
newid_082=[int(val/10) for val in fmli082['NEWID'].tolist()]
newid_083=[int(val/10) for val in fmli083['NEWID'].tolist()]
#check how large overlap is by getting
intersection=[val for val in newid_082 if val in newid_083]
#number of overlap 
overlap=len(intersection)
#share of households in third quarter that are in both (should be 75% as each quarter 25% are replaced (see CEX documentation))
#!only 65% (what is my mistake in this right now?)
share=overlap/len(fmli083)

#* STEP II: variables of (potential) relevance 
#create list of variables that have been part of survey in the relevant period, i.e. 2008 
#use the PUMD dictionary list of variables, second sheet in file
pumd_variables=pd.read_excel(data_in/'ce_pumd_interview_diary_dictionary.xlsx', sheet_name=1)
#if last year is missing variable is still part of survey 
#replace NaN in 'Last Year' with 2021 so it is not dropped in following steps
pumd_variables['Last year'][pumd_variables['Last year'].isnull()]=2021
#only keep variables which have "First Year" either before or in 2008
relevant_vars=pumd_variables[pumd_variables['First year']<=2008]
#also filter that last year is at least 2008 or in 2008
relevant_vars=relevant_vars[relevant_vars['Last year']>=2008]
#FPAR and MCHI datasets are not relevant as they only contain metadata 
relevant_vars=relevant_vars[(relevant_vars['File']!='MCHI')]
relevant_vars=relevant_vars[(relevant_vars['File']!='FPAR')]
#also don't need the single imputation iterations but only their average
#drop all rows that contain 'Imputation Iteration' in their variable description 
relevant_vars=relevant_vars[~relevant_vars['Variable description'].str.contains('Imputation Iteration')]
#also drop all rows that are an "Allocation Number"
#! what are those?? 
relevant_vars=relevant_vars[~relevant_vars['Variable description'].str.contains('Allocation Number')]
print(f'Left with {len(relevant_vars)} variables')
#also, filter section description to get a df containing only member characteristics and income 
#drop all rows that do not contain 'characteristics' in their section description 
relevant_chars=relevant_vars[relevant_vars['Section description'].str.contains('characteristics')]
#since here only interested in characteristics, let's ignore any variables that contain:
#'tax', 'income', 'clothing', 'expenditures', 'mortgage', 'expenses', 'food', 'beverages', 'tobacco', 'records', 'mortgage'
leave_outs=['tax', 'income', 'clothing', 'expenditures', 'mortgage', 'expenses', 'food', 'beverages', 'tobacco', 'records', 'mortgage', 'pay']
for var in leave_outs:
    relevant_chars=relevant_chars[~relevant_chars['Variable description'].str.contains(var)]
print(f'Left with {len(relevant_chars)} variables')

#filter by diary and interview
interview_relevant_chars=relevant_chars[relevant_chars['Survey']=='INTERVIEW']
diary_relevant_chars=relevant_chars[relevant_chars['Survey']=='DIARY']
#write filtered data to csv 
interview_relevant_chars.to_csv(data_out/'relevant_chars_interview.csv')
diary_relevant_chars.to_csv(data_out/'relevant_chars_diary.csv')