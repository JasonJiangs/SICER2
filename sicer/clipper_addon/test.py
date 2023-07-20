import numpy as np
from clipper import clipper

# Set random seed for reproducibility
np.random.seed(1)

n = 1000
trueidx = np.arange(0.1 * n)

score_exp = np.random.normal(size=n)
score_exp[:int(0.1 * n)] = np.random.normal(loc=5, size=int(0.1 * n))

# Inserting NA values randomly into score_exp
na_indices = np.random.choice(int(0.1 * n), 10, replace=False)
score_exp[na_indices] = np.nan

score_back = np.random.normal(size=n)

# Inserting NA values randomly into score_back
na_indices = np.random.choice(int(0.1 * n), 10, replace=False)
score_back[na_indices] = np.nan

FDR = np.array([0.001, 0.01, 0.05, 0.1])
aggregation_method = 'mean'
contrastScore_method = 'diff'
FDR_control_method = 'BC'
ifpowerful = True

# # Assuming 'clipper_addon' and all dependent functions are already defined in Python
# re_clipper = clipper_addon(score_exp,
#                      score_back,
#                      FDR=FDR,
#                      aggregation_method=aggregation_method,
#                      contrastScore_method=contrastScore_method,
#                      FDR_control_method=FDR_control_method,
#                      ifpowerful=ifpowerful)
