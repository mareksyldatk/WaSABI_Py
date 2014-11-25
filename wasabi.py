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
[ ] simplify to support only one kernel and one prior type, but multiple transformations
[ ] test from command line
[ ] verify with mc
[ ] Add **kwargs to likelihood computation
[ ] Add setseed
"""

from __future__ import division
import GPy
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt
from DIRECT import solve

'''
SUPPORT FUNCTIONS
-----
Set of basic support functions used in the rest of the code.
'''

# Reshape vector to column/row:
def to_column(x):
    if (x.__class__ == np.ndarray):
        return(x.reshape(-1,1))
    elif (x.__class__ == list):
        return(np.array([x]).reshape(-1,1))
    else:
        return(np.array([[x]]).reshape(-1,1))

def to_row(x):
    if (x.__class__ == np.ndarray):
        return(x.reshape(1,-1))
    elif (x.__class__ == list):
        return(np.array([x]).reshape(1,-1))
    else:
        return(np.array([[x]]).reshape(1,-1))
        
# Multivariable Normal PDF:    
def mvn_pdf(X, mean, cov):
    X, mean     = to_column(X), to_column(mean)
    k           = len(cov)
    den = np.sqrt( (2.0*np.pi)**k * np.linalg.det(cov))
    nom = np.exp( -0.5 * ((X-mean).T).dot(np.linalg.solve(cov, (X-mean))) )
    return( (nom/den)[0][0] )

# Covariance matrix symmetrizations:    
def symetrize_cov(P):
    return (P + P.T)/2.0
    
def mrdivide(B,A):   
    ''' Solves x A = B ==> x = B A^{-1}
        x = ( (A^T)^{-1} B^T )^T 
        Corresponds to matlabs B/A
    '''
    x_solve = (np.linalg.solve(A.T,B.T)).T
    # x_lstsq = (np.linalg.lstsq(A.T,B.T)).T
    return(x_solve)
'''
QUADRATURE CLASS
-----
Code defining BQ class
'''

#%% Define BQ Class
class BQ(object):
    """ Bayesian Quadrature class """

    # Constructor:
    def __init__(self, 
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
        self.dim                = None
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
        # Set optional hyper-param optimization/regression/prediction parameters:
        self.par_optimization   = {"num_restarts": 16, "verbose": False}
        self.par_regression     = {}
        self.par_prediction     = {}
        
        # LIKELIHOOD FUNCTION:
        # ----------
        self.likelihood_l = likelihood_l

        # PRIOR:
        # ----------
        # Prior type:
        self.prior_type         = prior_type  
        # Set default prior parameters depending on prior type:
        self.prior_parameters   = prior_parameters     
        
        # PyDIRECT SOLVE PARAMETERS:
        # ----------
        self.par_solve =  {"algmethod": 1, "maxT": 100, "maxf": 100}    
    
    # %%
    #
    # ##### ##### #####     METHODS: GAUSSIAN PROCESS     ##### ##### ##### #
    #
    
    def gp_regression(self, X, Y, **kwargs):
        """ Fit the Gaussian Process.
            Method uses GPy.models.GPRegression() to fit the GP model, using
            previously defined kernel and selected transformation. """
        # Default parameters:    
        if len(kwargs) == 0:
            kwargs = self.par_regression
        
        # Apply transformation
        if self.transformation == "wsabi-l":
            newY, self.alpha = self.sqrt_transform(Y)  
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

        # Optimize only if number of optimization iterations is set:
        if (self.par_optimization['num_restarts'] != 0):            
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
            mean, cov, lower, upper = self.sqrt_transform_inv(mean, cov)
            
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
            self.prior_parameters = {"mean": to_row(0.00), "cov": np.diag([1.00])}
        
    
    # METHOD: Get N samples from prior:
    # --    
    def sample_prior(self, N=1):
        """ Draw N samples from the prior. """
        # Normal prior (default):
        if self.prior_type == "normal":
            mean    = self.prior_parameters['mean'][0]
            cov     = self.prior_parameters['cov']
            samples = np.random.multivariate_normal(mean, cov, N)
            
        return (samples)
    
    # METHOD: Evaluate prior at given X
    # --
    def evaluate_prior(self, X):
        """ Evaluate value of prior at given point. """
        # Normal prior (default):
        if self.prior_type == "normal":
            mean   = self.prior_parameters['mean'][0]
            cov    = self.prior_parameters['cov']
            result = np.apply_along_axis(mvn_pdf, 1, X, mean, cov)
            
        return(result)
    # %%
    #
    # ##### ##### #####     METHODS: LIKELIHOOD     ##### ##### ##### #
    #
    
    # METHOD: Evaluate likelihood at given X
    # -- 
    def evaluate_likelihood(self, X):
        """ Evaluate likelihood on given X. """
        Y = np.apply_along_axis(self.likelihood_l, 1, X)
        return(Y)
        
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
        # self.X = self.sample_prior(1) if Xinit is None else Xinit   
        self.X = self.prior_parameters['mean'] if Xinit is None else Xinit  
        
        # Set initial observation(s)
        # Evaluated on X or set using Yinit
        self.Y = self.evaluate_likelihood(self.X) if Yinit is None else Yinit   
        
        # INITIALIZE GP
        # --
        # Fit GP:
        self.gp_regression(self.X, self.Y)
        # Set constraints on GP using dedicated function:
        if self.gp_set_constraints is not None:
            self.gp_set_constraints(self.gp)
        # Optimize new data:    
        self.gp_optimize()
        
        # SET DIM:
        # --
        self.dim = self.gp.input_dim
    
    # METHOD: Optimization objective
    # --        
    def opt_objective(self, X, return_zero=True):
        """ Optimization objective for DIRECT """
        X = to_row(X)
        # TODO: what happens to tilde_mean in multidim case??
        tilde_mean, tilde_cov, _ , _ = self.gp_prediction(X)
        # cost = ( self.evaluate_prior(X)**2 ) * tilde_cov * ( tilde_mean**2 )
        cost = ( self.evaluate_prior(X)**2 ) * tilde_cov * ( np.dot(tilde_mean, tilde_mean.T) )
        if return_zero:
            return( -cost , 0 )
        else:
            return( -cost )

    # METHOD: Predict next sample location
    # --    
    def find_next_sample(self):            
        """ Find lcoation of next sample """
        
        # Optimization range:
        if self.prior_type == "normal":
            mean = self.prior_parameters['mean']
            cov  = self.prior_parameters['cov']
            # TODO: Check if picking diag is OK
            lower_const = mean - 6.0*np.sqrt(cov.diagonal())
            upper_const = mean + 6.0*np.sqrt(cov.diagonal())
            
        # Wrap the optimization objective to use it within solve:    
        def mod_opt_obj(X, self):
            return(self.opt_objective(X))
            
        # Optimize: search for new location   
        # For 1 dimensionl input use grid search
        if (self.dim == 1):
            # Use grid:
            #TODO: Make it adjustable
            GRID_SIZE = 2500
            GRID_STEP = 0.01
            # Generate grid:
            X_grid = np.linspace(lower_const[0], upper_const[0], GRID_SIZE)
            #X_grid = np.arange(lower_const[0], upper_const[0], GRID_STEP)
            X_grid = to_column(X_grid)
            # Calculate objective:
            objective = np.apply_along_axis(self.opt_objective, 1, X_grid, False)
            objective = objective.tolist()
            
            # Pick X that maximizes the objective:
            max_ind = objective.index(min(objective)) # min since -cost         
            Xstar   = np.array([X_grid[max_ind]])    
        else:
            # Use DIRECT:
            kwargs = self.par_solve
            Xstar, _, _ = solve(mod_opt_obj, 
                                lower_const,
                                upper_const,
                                user_data=self, 
                                **kwargs)     
        # Assign result:
        self.Xstar = to_row(Xstar)
        print("Predicted new sample (Xstar): " + str(Xstar))

    # METHOD: Sample N samles
    # --
    def sample(self, N = 1):
        """ Sample N times """
        for n in range(0,N):
            # first sampling iteration
            if self.Xstar is None:
                # Get new sample
                self.find_next_sample()
            
            self.Ystar = self.evaluate_likelihood(self.Xstar)
                
            # Update X and Y
            self.X = np.vstack((self.X, self.Xstar))
            self.Y = np.vstack((self.Y, self.Ystar))      
            
            # Refit the model
            self.gp_regression(self.X, self.Y)
            if self.gp_set_constraints is not None:
                self.gp_set_constraints(self.gp) 
            self.gp_optimize()
            
            # Update Xstar
            self.Xstar = None
    
    # %%
    #
    # ##### ##### #####     METHODS: INTEGRAL     ##### ##### ##### #
    #
    # METHOD: Support function computing z for normal prior
    # --
    def compute_z(self, a, A, b, B, I, w_0):
        ''' Computes z for closed form solution integral '''
        # Make sure of column vectors:
        a, b = to_column(a), to_column(b)
        # Compute z:
        denominator = np.sqrt(np.linalg.det((np.linalg.solve(A,B)+I)))
        z = (w_0/denominator) * np.exp(-.5*((a-b).T).dot(np.linalg.solve((A+B),(a-b))))   
        return(z[0][0])
        
    # METHOD: Compute integral
    # --
    def compute_integral(self):
        ''' Compute mean and variance of the integral '''
        # For a normal prior:
        if self.prior_type == 'normal':
            
            # For WASABI-L transformation:
            if self.transformation == "wsabi-l":
                E_int = V_int = None
                
            # For a case of no transformation:
            else:
                # Fitted GP parameters      
                w_0 = self.gp.rbf.variance.tolist()[0]
                w_d = np.power(self.gp.rbf.lengthscale.tolist(), 2)
        
                # Parameters
                A = np.diag(w_d)
                I = np.eye(self.dim)     
        
                # Prior
                prior_mean = self.prior_parameters['mean']
                prior_cov  = self.prior_parameters['cov']
                
                # Compute z:
                z = [self.compute_z(x, A, prior_mean, prior_cov, I, w_0) for x in self.X]
                z = to_column(np.array(z))
                K = self.gp.kern.K(self.X)
                K = symetrize_cov(K)
                
                # Compute mean and variance of integral
                # TODO: check if V_int works properly
                E_int = (z.T).dot( np.linalg.solve(K, self.Y) )
                V_int = w_0/np.sqrt(np.linalg.det( 2*np.linalg.solve(A, prior_cov) + I) ) - (z.T).dot(np.linalg.solve(K,z))
                
        # Return computed values:
        return(E_int, V_int)        
        
    
    
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
        if (self.dim == 1):
            plt.figure()
            # PLOT 1:
            # Fitted GP mean(x) and variance(x), prior pi(x) and likelihood l(x)
            # Also: current samples (orange) and next sample (red)
            ax = plt.subplot2grid((3,3), (0, 0), colspan=3, rowspan=2)
            plt.rcParams['lines.linewidth'] = 1.5
            # Get scale
            if self.prior_type == "normal":
                mean = self.prior_parameters['mean']
                cov  = self.prior_parameters['cov']
                x_min = mean - 6.0*np.sqrt(cov.diagonal())
                x_max = mean + 6.0*np.sqrt(cov.diagonal())
                
            # Squeeze x_min and x_max for linspace    
            x_min = x_min.squeeze()
            x_max = x_max.squeeze()
            # Plot GP
            pltX  = to_column(np.linspace(x_min,x_max,1000))
            
            mean, cov, lower, upper = self.gp_prediction(pltX)
            
            GPy.plotting.matplot_dep.base_plots.gpplot(pltX, mean, lower, upper, ax=ax)
            # Plot likelihood:
            plt.plot(pltX, self.evaluate_likelihood(pltX), '-g')  
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
            return_zero = False
            pltY  = -np.apply_along_axis(self.opt_objective, 1, pltX, return_zero)
            self.pltY = pltY
            pltY[0]  = 0
            pltY[-1] = 0
            plt.plot(pltX, pltY, 'k', alpha=0.5)
            plt.fill(pltX, pltY, color='k', alpha=0.25)
            # Plot next sample:
            if self.Xstar is not None:
                plt.axvline(x=self.Xstar, color='r')
            # Change axis limits:
            plt.xlim([x_min, x_max])
        else:
            #TODO: Add multidimensional (at least 2) plotting option:
            print("Plot only supported for 1 dimensional input!")


    # %%
    #
    # ##### ##### #####     METHODS: WSABI-L     ##### ##### ##### #
    #
    
    # METHOD: Square root transform
    # --  
    # Applicable only for likelihoods
    def sqrt_transform(self, l_x):
        #    alpha = 0.5*np.min(l_x)
        #    tilde_l = np.sqrt(2*(l_x-alpha))
        #    return (tilde_l, alpha)  
        alpha   = 0.8 * l_x.min(axis=0)
        tilde_l = np.sqrt(2*(l_x-alpha))
        return (tilde_l, alpha)  
        
    # METHOD: Inverse sqrt transform
    # --  
    # Applicable only to likelihoods:
    def sqrt_transform_inv(self, tilde_mean, tilde_cov):
        #TODO: Check the inverse transformation with the paper for mdim case
        #TODO: Make it suitalbe for pair (x,x') - grant acces to gp
    
        # mean  = alpha + 0.5*tilde_mean**2
        # cov   = tilde_mean*tilde_cov*tilde_mean 
        
        mean = None
        cov  = None
        
        for i in range(0, len(tilde_mean)):
            mean_row = self.alpha + 0.5*tilde_mean[i]**2 
            cov_row  = tilde_cov[i]* np.dot(tilde_mean[i], tilde_mean[i].T)
            mean = mean_row if i == 0 else np.vstack((mean, mean_row))
            cov  = cov_row  if i == 0 else np.vstack((cov,  cov_row))
        
        # Lower and upper bounds only for 1 dim scenario (used for plot)
        if (self.dim == 1):
            lower = mean - 1.96*np.sqrt(cov)
            upper = mean + 1.96*np.sqrt(cov)
        else:
            lower = upper = None
            
        return(mean, cov, lower, upper)    
        
    # %%
    #
    # ##### ##### #####     METHODS: Simple Monte Carlo     ##### ##### ##### #
    #      
    def monte_carlo(self, N = 1000):
        # Sample prior:
        mc_X = self.sample_prior(N)
        mc_Y = self.evaluate_likelihood(mc_X)
        E_int = np.mean(mc_Y, axis=0)
        V_int = np.cov(mc_Y, rowvar=0)
        return(E_int, V_int)

        
#%%
#
# ##### ##### #####     TEST THE SCRIPT     ##### ##### ##### #
#
if __name__ == "__main__":
    import wasabi as wasabi

    #
    # SET UP EVERYTHING
    # 
    
    """ LIKELIHOOD: Define a likelihood l(x) function, used in Bayesian 
        Quadrature. Likelihood is a function of X and returns Y.
        Both input X and output Y are assumed row arrays, where each line 
        represents one vector.
    """
    def likelihood_fcn_one_dim(x):
        #TODO: fix this squeeze
        x = x.squeeze()
        par = {'delta': 0.5, 'mu': 0.0, 'sigma': 0.75, 'A': 0.5, 'phi': 10.0, 'offset': 0.0}    
        # Compute components of y:
        y_a = (1.0 - np.exp(par['delta']*x)/(1+np.exp(par['delta']*x)))
        y_b = np.exp(-(x-par['mu'])**2 / par['sigma'])
        y_c = par['A']* np.cos(par['phi']*x)
        # Return y:
        y = y_a + y_b * y_c + par['offset']
        y = to_row(y)
        return(y)
        
    def likelihood_fcn_multi_dim(x):
        par = {'delta': 0.5, 'mu': 0.0, 'sigma': 0.75, 'A': 0.5, 'phi': 10.0, 'offset': 0.0}    
        # Compute components of y:
        y_a = (1.0 - np.exp(par['delta']*x[0])/(1+np.exp(par['delta']*x[0])))
        y_b = np.exp(-(x[1]-par['mu'])**2 / par['sigma'])
        y_c = par['A']* np.cos(par['phi']*x[1])
        # Return y:
        y = y_a + y_b * y_c + par['offset']
        y = to_row(y)
        return(y)
        
    """ PRIOR: Set parameters for default ("normal") prior. If not specified, 
        parameters are set to default ({"mean": 0.00, "cov": 1.00})
    """
    prior_mean = to_row([0])
    prior_cov  = np.diag([1])
    prior_parameters = {"mean": prior_mean, "cov": prior_cov}
    
    """ GP KERNEL: Set up kernel for the Gaussian Process. Kernel is defined in
        the same manner as in GPy.
        NOTE: Currently only supported rbf kernel!
    """    
    gp_kernel = GPy.kern.RBF(input_dim=1, ARD=True)
    
    """ GP OPTMIMIZATION CONSTRAINTS: Set constraints for the GP optimization.
        Set the same as for GPy.
        NOTE: Currently only supported parameters for rbf kernel!
    """  
    def gp_set_constraints(gp):
        gp.unconstrain('')
        gp.rbf.variance.constrain_positive(warning=False)
        gp.rbf.lengthscale.constrain_bounded(0.1, 5.0,warning=False)
        #gp.rbf.variance.constrain_fixed(1.0, warning=False)        
        #gp.rbf.lengthscale.constrain_fixed(1.0, warning=False)
        gp.Gaussian_noise.variance.constrain_fixed(0.0001**2, warning=False)
        
        return(gp)
    
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
    bqm = wasabi.BQ(likelihood_l        = likelihood_fcn_one_dim, 
                    gp_kernel           = gp_kernel,
                    gp_set_constraints  = gp_set_constraints,
                    prior_parameters    = prior_parameters,
                    transformation      = 'none')
                    
    bqm.par_optimization['num_restarts'] = 4
    
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
    bqm.sample(N=64)  
    
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
    
    """ COMPUTE INTEGRAL: Using quadrature and Monte Carlo
    """
    E, V        = bqm.compute_integral()
    E_mc, V_mc  = bqm.monte_carlo(10000)
    
    print(E.squeeze(), E_mc.squeeze())
    print(V, V_mc)