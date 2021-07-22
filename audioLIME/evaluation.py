import numpy as np
from scipy.stats import entropy

def compute_complexity(explanation, target_idx):
    local_exp = explanation.local_exp[target_idx]
    absolute_weights = np.array([np.abs(x[1]) for x in local_exp])
    sum_weigths = sum(absolute_weights)
    Pg = np.array([gi / sum_weigths for gi in absolute_weights])
    return entropy(Pg, base=len(Pg))
