# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:27:02 2014

@author: marek
"""
from __future__ import division
import GPy
import numpy as np
import scipy as sp
import wasabi_support as ws
import matplotlib.pyplot as plt
from DIRECT import solve

#%% Define BQ Class
class BQ(object):
    """ This is a Bayesian Quadrature class """

    # Constructor:
    def __init__(self, dim=1, obs_noise=0.0001, likelihood_l=None, 
                 prior_type = "normal", 
                 transformation="square-root", 
                 approximation="linearisation", 
                 sampling="uncertainty"
                 ):
                     
        # Basic settings:
        self.dim            = dim
        self.transformation = transformation
        self.approximation  = approximation
        self.sampling       = sampling
        
        self.alpha = None
        self.obs_noise = obs_noise
        
        # GPy parameters:        
        self.gp = None
        self.gp_kernel = None
        # GPy optimization and regression settings:
        self.gp_parOptimization = {"num_restarts": 10, "verbose": False}
        self.gp_parRegression   = {}
        self.gp_parPrediction   = {}
        
        # Likelihood function
        self.likelihood_l = likelihood_l
        
        # Prior parameters
        self.prior_type = prior_type  
        self.prior_parameters = None     
        self.set_prior_parameters()     # Set default prior parameters
        
        # Data points
        self.X = None
        self.Y = None
        # Next sample point
        self.Xstar = None
        self.Ystar = None        
        
    # Methods: PRINT DETAILS    
    def details(self):
        print("\nBayesian Quadrature model details:")
        print("- "*10)
        print("Input dimension: "      + str(self.dim))
        print("Transformation type: "  + self.transformation)
        print("Approximation method: " + self.approximation)        
        print("Sampling method: "      + self.sampling)   
        print("\nPrior type: "         + self.prior_type)
        print("Priot parameters: "), 
        print(self.prior_parameters)
        print("\nGP details:")
        print("- "*10),
        print(self.gp)
    
    # Methods: GAUSSIAN PROCESS
    def gp_regression(self, X, Y, **kwargs):
        """ Fit the Gaussian Process.
            Method uses GPy.models.GPRegression() to fit the GP model, using
            previously defined kernel. """
        # Default parameters:    
        if len(kwargs) == 0:
            kwargs = self.gp_parRegression
        
        # Apply transformation
        if self.transformation == "square-root":
            #print("transform")
            newY, self.alpha = ws.log_transform(Y)
        else:
            newY = Y
        # Fit GP model        
        self.gp = GPy.models.GPRegression(X, newY, self.gp_kernel, **kwargs)

    def gp_optimize(self, **kwargs):
        """ Optimize GP hyperparameters.
            Method uses GPy.optimize_restarts(), to optimize the hyper 
            parameters. """
        if len(kwargs) == 0:
            kwargs = self.gp_parOptimization
            
        self.gp.optimize_restarts(**kwargs)
        
    def gp_prediction(self, Xnew, **kwargs):
        """ GP prediction of mean, variance and confidence interval. Function
            uses different predefined transformations. """
        # Predict mean and covariance
        mean, cov = self.gp.predict(Xnew, **kwargs)
        # Predict quantiles
        lower, upper = self.gp.predict_quantiles(Xnew)
        
        # Invert the transformation:
        if self.transformation == "square-root":
            # print("inv transform")
            mean, cov, lower, upper = ws.log_transform_inv(mean, cov, self.alpha)
            
        # Return all
        return(mean, cov, lower, upper)
        
    def plot(self):
        # PLOT 1:
        # Fitted GP mean(x) and variance(x), prior pi(x) and likelihood l(x)
        # Also: current samples (orange) and next sample (red)
        ax = plt.subplot(211)
        plt.rcParams['lines.linewidth'] = 1.5
        # Get scale
        if self.prior_type == "normal":
            mu = self.prior_parameters['mu']
            sigma = self.prior_parameters['sigma']
            x_min = mu - 3*sigma
            x_max = mu + 3*sigma
        
        # Plot GP
        pltX  = np.array([np.linspace(x_min,x_max,1000)]).T
        mean, cov, lower, upper = self.gp_prediction(pltX)
        GPy.plotting.matplot_dep.base_plots.gpplot(pltX, mean, lower, upper, ax=ax)
        # Plot likelihood:
        plt.plot(pltX, self.likelihood_l(pltX), '-r')  
        # Plot prior:
        plt.plot(pltX, self.evaluate_prior(pltX), color='#ffa500')
        # Plot observations:
        plt.plot(self.X, self.Y, 'o', color='#ffa500', ms=5)
        # Plot next sample:
        if self.Xstar is not None:
            plt.plot(self.Xstar, self.Ystar, 'o', color='r', ms=5)
        # Change axis limits:
        plt.xlim([x_min, x_max])
        
        # PLOT 2:
        # Plot objective
        ax = plt.subplot(2,1,2)
        pltY  = -self.opt_objective(pltX)[0]
        pltY[0]  = 0
        pltY[-1] = 0
        plt.plot(pltX, pltY, 'k', alpha=0.5)
        plt.fill(pltX, pltY, color='k', alpha=0.25)
        # Change axis limits:
        plt.xlim([x_min, x_max])
        return None
        
    # Methods: PRIOR FUNCTIONS
    def set_prior_parameters(self, **kwargs):
        """ Set prior parameters.
            If no **kwargs given, set defaults based on prior type."""
            
        # Normal prior (default)
        if self.prior_type == "normal":
            if len(kwargs) != 0:
                self.prior_parameters = kwargs
            else:
                self.prior_parameters = {"mu": 0.0, "sigma": 1.00}
        
    def sample_prior(self, N=1):
        """ Draw N samples from the prior. """
        # Normal prior (default):
        if self.prior_type == "normal":
            mu      = self.prior_parameters['mu']
            sigma   = self.prior_parameters['sigma']
            samples = np.random.normal(mu, sigma, N)
            # Convert to numpy array
            samples = np.array([[s] for s in samples])
        return (samples)
    
    def evaluate_prior(self, X):
        """ Evaluate value of prior at given point. """
        # Normal prior (default):
        if self.prior_type == "normal":
            mu     = self.prior_parameters['mu']
            sigma  = self.prior_parameters['sigma']
            result = sp.stats.norm.pdf(X, mu, sigma)
            
        return(result)
        
    # Methods: QUADRATURE
    def initialize_sampler(self, Xinit=None):
        """ Initialize sampler with first sample. 
            Fit GP immediately. """
        
        # Get initial sample (randomly sampled from prior or set using Xinit)
        if Xinit is None:
            self.X = self.sample_prior(1)
        else:
            self.X = Xinit
            
        # Prepare observations:
        self.Y = self.likelihood_l(self.X)

        # Fit GP and optimize hyperparams
        self.gp_regression(self.X, self.Y)
        self.gp_optimize()
            
    def opt_objective(self, X=None):
        """ Optimization objective for DIRECT """
        if X is None:
            X = self.X
        tilde_mean, tilde_cov, _ , _ = self.gp_prediction(X)
        cost = ( self.evaluate_prior(X)**2 ) * tilde_cov * ( tilde_mean**2 )
        return( -cost , 0 )

        
    def predict_sample(self):            
        """ Active sampler based on chosen method """
        # Optimization range:
        if self.prior_type == "normal":
            mu = self.prior_parameters['mu']
            sigma = self.prior_parameters['sigma']
            lower_const = mu - 3*sigma
            upper_const = mu + 3*sigma
        # Wrap the function:    
        def mod_opt_obj(X, self):
            return(self.opt_objective(np.array([X])))
        # Optimize    
        Xstar, _, _ = solve(mod_opt_obj, lower_const, upper_const, user_data=self, algmethod = 1,
                                 maxT = 333, maxf = 1000)
        return(Xstar)  

    def sample(self, N=1):
        for n in range(0,N):
            # first sampling iteration
            if self.Xstar is None:
                # Get new sample
                self.Xstar = self.predict_sample()
                self.Ystar = self.likelihood_l(self.Xstar)
                
            # Update X and Y
            self.X = np.append(self.X, self.Xstar)
            self.X = np.array([self.X]).T
            self.Y = np.append(self.Y, self.Ystar)
            self.Y = np.array([self.Y]).T
            # Refit the model
            self.gp_regression(self.X,self.Y)
            self.gp_optimize()
            
            # Get new sample
            Xstar = self.predict_sample()
            Ystar = self.likelihood_l(Xstar)
            # Update Xstar
            self.Xstar = Xstar
            self.Ystar = Ystar
        
        return(None)

#%% # ##### ##### ##### ##### #
#
#       TEST THE SCRIPT
#
##### ##### ##### ##### ##### #
if __name__ == "__main__":
    import WaSABI_Py as wasabi
    
    # Define likelihood    
    
    def likelihood_fcn(x):
        par = {'delta': 0.5, 'mu': 0.0, 'sigma': 0.75, 'A': 0.5, 'phi': 10.0, 'offset': 0.0}    
        # Compute components of y:
        y_a = (1.0 - np.exp(par['delta']*x)/(1+np.exp(par['delta']*x)))
        y_b = np.exp(-(x-par['mu'])**2 / par['sigma'])
        y_c = par['A']* np.cos(par['phi']*x)
        # Return y:
        y = y_a + y_b * y_c + par['offset']
        return(y)
    
    # Create quadrature object    
    bqm = wasabi.BQ(dim=1, likelihood_l=likelihood_fcn)
    # Set GP kernel    
    bqm.gp_kernel = GPy.kern.RBF(input_dim = 1)
    # Set prior    
    bqm.set_prior_parameters()

    # Initialize sampler (draw 1st sample)
    #Xinit = np.array([np.linspace(-7.5,7.5,150)]).T
    #Xinit = np.array([[0]])
    Xinit = None
    bqm.initialize_sampler(Xinit=Xinit)
    
    # Set boundaries on parameters and reoptimize
    bqm.gp.unconstrain('')
    bqm.gp.rbf.variance.constrain_positive()
    bqm.gp.rbf.lengthscale.constrain_bounded(0.0,.25)
    bqm.gp.Gaussian_noise.constrain_fixed(bqm.obs_noise**2)
    bqm.gp_optimize()
    # Get samples using quadrature:
    bqm.sample(N=10)

    # Print details
    #bqm.details()  
    bqm.plot()    
    
#    # How to use different methods:
#    # ----------    
#    # Data points:
#    X = np.array([[-2.0],[0.0], [2.0]])
#    Y = np.array([ [4.0],[0.0], [4.0]])
#    # Regression
#    bqm.gp_regression(X,Y)
#    # Optimize
#    bqm.gp_optimize()