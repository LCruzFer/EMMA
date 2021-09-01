# Master Thesis: Heterogeneity in MPC
## General
This repository contains code for data pre-processing, analysis and visualization for my Master's thesis on heterogeneity in the Marginal Propensity to Consume (MPC). I use a Double/Debiased Machine Learning (DML) - also known as Orthogonal Machine Learning - approach to identify the effect of the 2008 US tax stimulus on households' consumption. This allows me to uncover not pre-defined patterns of heterogeneity. At the same time, contrary to several seminal contributions in the literature, the estimator of my choice allows for conditional sequential exogeneity and hence I can control for lagged values of consumption and the treatment - the tax rebate received - without leading to inconsistency of my results. 

## Structure 
### Python Code
The src directory contains the python code for data pre-processing, estimation and visualization.

Data is prepared in the file *prep_data.py*. *utils.py* contains classes that contain methods for data cleaning, splitting and estimation that are used across multiple files of the analysis. 

The sub-directory *analysis.py* contains the files for the estimation procedure. In *tune_first_stage.py* the hyperparameters of the Random Forest (first stage of DML) are tuned. Estimation of baseline Linear DML is done in *linearDML_baseline.py*. 

Finally, code for any figures generated can be found in the figures subdirectory.

### LaTeX & Figures
Any latex files containing files for discussions or drafts of the thesis can be found in docs. The figures generated in src/figures are saved as pdf/jpg in the figures directory. 

**Note that all written drafts are still subject to major changes, contain possibly incorrect explanations and quick thoughts I had.**

## References 
Coming soon...