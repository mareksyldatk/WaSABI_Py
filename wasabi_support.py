# -*- coding: utf-8 -*-
"""
Set of support functons for WaSABI_Py package

@author: Marek Syldatk
"""
import numpy as np
import sys, os

#%% DISABLE/RESTORE output
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    

# WASABI SUPPORT FUCTIONS

# Log transform:
def log_transform(l_x):
    alpha = 0.5*np.min(l_x)
    tilde_l = np.sqrt(2*(l_x-alpha))
    return (tilde_l, alpha)  

# Inverse lof transform:
def log_transform_inv(tilde_mean, tilde_cov, alpha):
    mean  = alpha + 0.5*tilde_mean**2
    cov   = tilde_mean*tilde_cov*tilde_mean    
    lower = mean - 1.96*np.sqrt(cov)
    upper = mean + 1.96*np.sqrt(cov)
    return(mean, cov, lower, upper)    
    
