# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 12:08:41 2014

@author: marek
"""
import numpy as np

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