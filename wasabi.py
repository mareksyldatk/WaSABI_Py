# -*- coding: utf-8 -*-
"""
WaSABI_Py Bayesian Quadrature package

@author: Marek Syldatk
@references:
[1]
[2]

TODO Notes:
[ ] sample() and initialization() - remove model refitting, compact the 
    repeated code int oone method to avoid problems
[ ] add mean_Z() and var_Z(): methods to compute mean and var of the integral
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
    """ Bayesian Quadrature class """

    # Constructor:
    def __init__(self, 
                 dim                = 1, 
                 obs_noise          = 0.001, 
                 likelihood_l       = None, 
                 gp_kernel          = None,
                 gp_set_constraints = None,
                 prior_type         = "normal", 
                 prior_parameters   = None,
                 transformation     = "wsabi-l", 
                 sampling           = "uncertainty"
                 ):
                     
        # BASIC SETTINGS:
        # ----------
        # Input dimension
        self.dim                = dim
        # Set transformation/approximation/sampling type:
        self.transformation     = transformation
        self.sampling           = sampling
        # Set observation noise:
        self.obs_noise          = obs_noise
        
        # DATA POINTS:
        # ---------
        # Current sample/likelihood values:
        self.X      = None
        self.Y      = None
        # Predicted sample value:
        self.Xstar  = None
        
        # TRANSFORMATION PARAMETERS:
        # ----------         
        # Square-root transformation:
        self.alpha = None
        
        # GAUSSIAN PROCESS PARAMETERS:
        # ----------
        # Set GP object and kernel
        self.gp                 = None
        self.gp_kernel          = gp_kernel
        # Set GP parameter constraints using function:
        self.gp_set_constraints = gp_set_constraints
        # Set optional optimization/regression/prediction parameters:
        self.par_optimization   = {"num_restarts": 10, "verbose": False}
        self.par_regression     = {}
        self.par_prediction     = {}
        
        # LIKELIHOOD FUNCTION:
        # ----------
        self.likelihood_l = likelihood_l

        # PRIOR:
        # ----------
        # Prior type:
        self.prior_type         = prior_type  
        # Set default prior parameters dependint on prior type:
        self.prior_parameters   = prior_parameters     
        
        # PyDIRECT SOLVE PARAMETERS:
        # ----------
        self.par_solve =  {"algmethod": 1, "maxT": 333, "maxf": 1000}    
    
    # %%
    #
    # ##### ##### #####     METHODS: GAUSSIAN PROCESS     ##### ##### ##### #
    #
    
    def gp_regression(self, X, Y, **kwargs):
        """ Fit the Gaussian Process.
            Method uses GPy.models.GPRegression() to fit the GP model, using
            previously defined kernel. """
        # Default parameters:    
        if len(kwargs) == 0:
            kwargs = self.par_regression
        
        # Apply transformation
        if self.transformation == "wsabi-l":
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
            kwargs = self.par_optimization
            
        self.gp.optimize_restarts(**kwargs)
        
    def gp_prediction(self, Xnew, **kwargs):
        """ GP prediction of mean, variance and confidence interval. Function
            uses different predefined transformations. """
        # Predict mean and covariance
        mean, cov = self.gp.predict(Xnew, **kwargs)
        # Predict quantiles
        lower, upper = self.gp.predict_quantiles(Xnew)
        
        # Invert the transformation:
        if self.transformation == "wsabi-l":
            mean, cov, lower, upper = ws.log_transform_inv(mean, cov, self.alpha)
            
        # Return all
        return(mean, cov, lower, upper)
        
    # %%
    #
    # ##### ##### #####     METHODS: PRIOR     ##### ##### ##### #
    #
        
    # METHOD: Set default prior parameters:
    # --
    def set_default_prior_parameters(self):
        """ Set default prior parameters. """
            
        # Normal prior (default)
        if self.prior_type == "normal":
            self.prior_parameters = {"mu": 0.00, "sigma": 1.00}
        
    
    # METHOD: Get N samples from prior:
    # --    
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
    
    # METHOD: Evaluate prior at given X
    # --
    def evaluate_prior(self, X):
        """ Evaluate value of prior at given point. """
        # Normal prior (default):
        if self.prior_type == "normal":
            mu     = self.prior_parameters['mu']
            sigma  = self.prior_parameters['sigma']
            result = sp.stats.norm.pdf(X, mu, sigma)
            
        return(result)
        
    # %%
    #
    # ##### ##### #####     METHODS: QUADRATURE     ##### ##### ##### #
    #
    
    # METHOD: Sampler initialization
    # --    
    def initialize_sampler(self, Xinit=None, Yinit=None):
        """ Initialize sampler with first sample. 
            Fit GP immediately. """
        
        # Initialize prior with default parameters:
        if self.prior_parameters is None:
            self.set_default_prior_parameters()       
        
        # INITIAL DATA:
        # --
        # Set initial sample(s)
        # Randomly sampled from prior or set using Xinit
        self.X = self.sample_prior(1) if Xinit is None else Xinit        
        
        # Set initial observation(s)
        # Evaluated on X or set using Yinit
        self.Y = self.likelihood_l(self.X) if Yinit is None else Yinit   

        # INITIALIZE GP
        # --
        # Fit GP:
        self.gp_regression(self.X, self.Y)
        # Set constraints on GP using dedicated function:
        if self.gp_set_constraints is not None:
            self.gp_set_constraints(self.gp)
        # Optimize new data:    
        self.gp_optimize()
    
    # METHOD: Optimization objective
    # --        
    def opt_objective(self, X=None):
        """ Optimization objective for DIRECT """
        if X is None:
            X = self.X
        tilde_mean, tilde_cov, _ , _ = self.gp_prediction(X)
        cost = ( self.evaluate_prior(X)**2 ) * tilde_cov * ( tilde_mean**2 )
        return( -cost , 0 )

    # METHOD: Predict next sample location
    # --    
    def find_next_sample(self):            
        """ Find lcoation of next sample """
        
        # Optimization range:
        if self.prior_type == "normal":
            mu = self.prior_parameters['mu']
            sigma = self.prior_parameters['sigma']
            lower_const = mu - 3*sigma
            upper_const = mu + 3*sigma
            
        # Wrap the optimization objective to use it within solve:    
        def mod_opt_obj(X, self):
            return(self.opt_objective(np.array([X])))
            
        # Optimize: search for new location   
        kwargs = self.par_solve
        Xstar, _, _ = solve(mod_opt_obj, 
                            lower_const,
                            upper_const,
                            user_data=self, 
                            **kwargs)     
        # Assign result:
        self.Xstar = Xstar
        print("\nPredicted new sample (Xstar): " + str(Xstar) + "\n")

    # METHOD: Sample N samles
    # --
    def sample(self, N = 1):
        """ Sample N times """
        for n in range(0,N):
            # first sampling iteration
            if self.Xstar is None:
                # Get new sample
                self.find_next_sample()
            
            self.Ystar = self.likelihood_l(self.Xstar)
                
            # Update X and Y
            self.X = np.append(self.X, self.Xstar)
            self.X = np.array([self.X]).T
            self.Y = np.append(self.Y, self.Ystar)
            self.Y = np.array([self.Y]).T
            # Refit the model
            self.gp_regression(self.X, self.Y)
            if self.gp_set_constraints is not None:
                self.gp_set_constraints(self.gp) 
            self.gp_optimize()
            
            # Update Xstar
            self.Xstar = None
    
    # %%
    #
    # ##### ##### #####     METHODS: OUTPUT     ##### ##### ##### #
    #
    
    # METHOD: Print details  
    # --
    def details(self):
        print("\n\n\nBayesian Quadrature object details:")
        print("- "*25)
        print("Input dimension: "      + str(self.dim))
        print("Transformation type: "  + self.transformation)       
        print("Sampling method: "      + self.sampling)   
        print("\nPrior type: "         + self.prior_type)
        print("Priot parameters: "), 
        print(self.prior_parameters)
        print("\nNumber of samples sampled: " + str(len(self.X)))
        if self.Xstar is not None:
            print("Next sample: " + str(self.Xstar))
        print("\nGP details (from GPy):")
        print("- "*10),
        print(self.gp)
        
    # METHOD: Plot results
    # --
    def plot(self):
        # PLOT 1:
        # Fitted GP mean(x) and variance(x), prior pi(x) and likelihood l(x)
        # Also: current samples (orange) and next sample (red)
        ax = plt.subplot2grid((3,3), (0, 0), colspan=3, rowspan=2)
        plt.rcParams['lines.linewidth'] = 1.5
        # Get scale
        if self.prior_type == "normal":
            mu = self.prior_parameters['mu']
            sigma = self.prior_parameters['sigma']
            x_min = mu - 6*sigma
            x_max = mu + 6*sigma
        
        # Plot GP
        pltX  = np.array([np.linspace(x_min,x_max,1000)]).T
        mean, cov, lower, upper = self.gp_prediction(pltX)
        GPy.plotting.matplot_dep.base_plots.gpplot(pltX, mean, lower, upper, ax=ax)
        # Plot likelihood:
        plt.plot(pltX, self.likelihood_l(pltX), '-g')  
        # Plot prior:
        plt.plot(pltX, self.evaluate_prior(pltX), color='#ffa500')
        # Plot observations:
        plt.plot(self.X, self.Y, 'o', color='#ffa500', ms=5)
        # Plot next sample:
        if self.Xstar is not None:
            plt.axvline(x=self.Xstar, color='r')
        # Change axis limits:
        plt.xlim([x_min, x_max])
        plt.title("Bayesian Quadrature")
        
        # PLOT 2:
        # Plot objective
        ax = plt.subplot2grid((3,3), (2, 0), colspan=3)
        pltY  = -self.opt_objective(pltX)[0]
        pltY[0]  = 0
        pltY[-1] = 0
        plt.plot(pltX, pltY, 'k', alpha=0.5)
        plt.fill(pltX, pltY, color='k', alpha=0.25)
        # Change axis limits:
        plt.xlim([x_min, x_max])

        
#%%
#
# ##### ##### #####     TEST THE SCRIPT     ##### ##### ##### #
#
if __name__ == "__main__":
    import WaSABI_Py as wasabi

    #
    # SET UP EVERYTHING
    # 
    
    """ LIKELIHOOD: Define a likelihood l(x) function, used in Bayesian 
        Quadrature. Likelihood is a function of X and returns Y.
    """
    def likelihood_fcn(x):
        par = {'delta': 0.5, 'mu': 0.0, 'sigma': 0.75, 'A': 0.5, 'phi': 10.0, 'offset': 0.0}    
        # Compute components of y:
        y_a = (1.0 - np.exp(par['delta']*x)/(1+np.exp(par['delta']*x)))
        y_b = np.exp(-(x-par['mu'])**2 / par['sigma'])
        y_c = par['A']* np.cos(par['phi']*x)
        # Return y:
        y = y_a + y_b * y_c + par['offset']
        return(y)
        
    """ PRIOR: Set parameters for default ("normal") prior. If not specified, 
        parameters are set to default ({"mu": 0.00, "sigma": 1.00})
    """
    prior_parameters = {"mu": 0.10, "sigma": 1.5}
    
    """ GP KERNEL: Set up kernel for the Gaussian Process. Kernel is defined in
        the same manner as in GPy.
    """    
    gp_kernel = GPy.kern.RBF(input_dim=1)
    
    """ GP OPTMIMIZATION CONSTRAINTS: Set constraints for the GP optimization.
        Set the same as for GPy.
    """  
    def gp_set_constraints(gp):
        gp.unconstrain('')
        gp.rbf.variance.constrain_positive()
        gp.rbf.lengthscale.constrain_bounded(0.0, 1.0)
        gp.Gaussian_noise.variance.constrain_fixed(0.001**2)
    
    #
    # CREATE QUADRATURE OBJECT AND PERFORM SAMPLING
    #   

    """ BQ OBJECT: Pass mandatory parameters to create BQ object.
        ---
        
        PARAMETERS:
        - dim:                  Input dimension (default: 1)
        - obs_noise:            Observation noise std (default: 0.001)
        - likelihood_l:         Likelihood function (mandatory)
        - gp_kernel:            GP kernel from GPy (mandatory)
        - gp_set_constraints:   Function setting constraints for GP (optional)
        - prior_type:           Prior type (default: "normal")
        - prior_prameters:      Set parameters of the prior, if None then the
                                default ones for given prior type are set 
                                (optional)
        - transformations:      Transformation type (default: "wsabi-l")
        - sampling:             Sampling method (default: "uncertainty").                  
    """
    bqm = wasabi.BQ(likelihood_l        = likelihood_fcn, 
                    gp_set_constraints  = gp_set_constraints,
                    gp_kernel           = gp_kernel,
                    prior_parameters    = prior_parameters)

    """ SAMPLER INITIALIZATION: It can be done either randomly (no parameters, 
        initial sample sampled from prior) or using given values both for 
        initial Xinit and Yinit. If Yinit is not specified, it will be 
        computed using likelihood and Xinit. 
        
        Example:
        --
        Xinit = np.array([np.linspace(-7.5,7.5,15)]).T
    """
    bqm.initialize_sampler()
    
    """ SAMPLING: Sample N (default N=1) samples using defined sampling method. 
        Available options or now: 'uncertainty'. New samples and corresponding 
        likelihood values are stored as self.X and self.Y
    """
    bqm.sample(N=25)  
    
    """ FIND NEXT SAMPLE (optional): Use optimization to find new sample 
        location. New samples is stored as self.Xstar. This is optional, 
        if there is no next sample found, it will not be plotted.
    """    
    bqm.find_next_sample()
    
    """ PLOTTING: Plot everything.
    """
    bqm.plot()
    
    """ DETAILS: Print details about the BQ object.
    """
    bqm.details()