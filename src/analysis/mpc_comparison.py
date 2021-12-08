import time
import sys
from pathlib import Path
from typing import AsyncGenerator
from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#* set system path to import utils
wd=Path.cwd()
sys.path.append(str(wd.parent))
from utils import data_utils
from utils import fitDML
from ALEPython.src.alepython import ale_dml
#import ALEPython.src.alepython as ale_dml

#* set data paths
data_in=wd.parents[2]/'data'/'in'
data_out=wd.parents[2]/'data'/'out'
fig_out=wd.parents[2]/'figures'
results=wd.parents[2]/'data'/'results'

'''
This file compares effects across different consumption categories.
'''