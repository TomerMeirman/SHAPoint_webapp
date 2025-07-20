import numpy as np
import pandas as pd
from shapoint_webapp.model_manager import ModelManager

def test_trim_small_bins():
    cfg={'model':{'task':'Classification','k_variables':1,'max_leaves':2,'score_scale':10,'use_optuna':False,'n_random_features':0,'params_path':'','model_path':''},'data':{'default_dataset':'breast_cancer','data_path':'','data_separator':',','target_column':'target','exclude_columns':[],'feature_types':{}}}
    mm=ModelManager(cfg)
    # create synthetic scores 0-9, duplicates, plus one outlier 30
    scores=pd.Series([0]*100 + [1]*100 + [2]*100 + [3]*100 + [4]*100 + [5]*100 + [6]*100 + [7]*100 + [8]*100 + [9]*100 + [30])
    outcomes=pd.Series(np.random.randint(0,2,len(scores)))
    bins,_=mm._compute_risk_bins(scores,outcomes,n_bins=8)
    # last bin has 1 sample which equals threshold, so valid
    assert bins[7]['valid'] is True
    # a mid bin should be valid
    assert bins[0]['valid'] is True 