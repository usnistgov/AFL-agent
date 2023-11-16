import numpy as np
import gpflow
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard
)
import xarray as xr
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import tqdm
import warnings

from AFL.agent.util import ternary_to_xy

class HeteroscedasticGaussian(gpflow.likelihoods.Likelihood):
    """
    Defines the heteroscedastic gaussian likelihood 
    """
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError


class DummyGP:
    def __init__(self,dataset,kernel=None):
        self.dataset=dataset
        self.kernel=kernel
        
    def predict(self,components):
        domain = self.dataset.afl.comp.get(components)
        self.y_mean = np.zeros_like(domain)
        self.y_var = np.zeros_like(domain)
        return {'mean':self.y_mean,'var':self.y_var}

    
class GPR:
#   '''
#   This is a class for doing regression. 
#   -------------
#   The data construct needs to point to the values being regressed over, Y_i
#   If uncertainties are known for each value, set the heteroscedastic flag to True
#   
#   Pass the inputs and uncertainties into the initialization
#    '''    
#
    def __init__(self,dataset,inputs=None,uncertainties=None,kernel=None,heteroscedastic=False):
        self.kernel = kernel
        self.heteroscedastic = heteroscedastic
        self.reset_GP(dataset=dataset,kernel=kernel,heteroscedastic=heteroscedastic)
        self.iter_monitor = lambda x: None
        self.final_monitor = lambda x: None
        
    def construct_data(self,heteroscedastic=True, preprocess=True):
           
        domain = self.transform_domain()
        if heteroscedastic:
            inputs = self.dataset[self.dataset.attrs['AL_regression_data']]
            uncertainties = self.dataset[self.dataset.attrs['AL_regression_uncertainties']]
            targets = np.stack((inputs,uncertainties), axis=1)
            targets[:,0] = (targets[:,0] - targets[:,0].mean())/targets[:,0].std()
            targets[:,1] = targets[:,1]/targets[:,1].std()

        else:
            #targets = self.dataset(self.inputs)
            targets = xr.DataArray(np.expand_dims(np.array(self.dataset[self.dataset.attrs['AL_regression_data']].values),axis=1))
            targets = (targets - targets.mean())/targets.std()
        data = (domain,targets)
        return data
        

   # def transform_domain(self,components=None):
   ##     """
   ##     Transforms the evaluated space (i.e. compositions axes) to a range from 0-1. 
   ##     --------
   ##     """    
   #     if 'GP_domain_transform' in self.dataset.attrs:
   #         if self.dataset.attrs['GP_domain_transform']=='ternary':
   #             if not (len(self.dataset.attrs['components'])==3):   
   #                 raise ValueError("Ternary domain transform specified but len(components)!=3") 
   #             domain = self.dataset.afl.comp.ternary_to_xy(components=components)
   #         elif self.dataset.attrs['GP_domain_transform']=='standard_scaled':
   #             domain = self.dataset.afl.comp.get_standard_scaled(components=components)
   #         elif self.dataset.attrs['GP_domain_transform']=='range_scaled':
   #             components = self.dataset.afl.comp._get_default(components)
   #             ranges = {}
   #             for component in components:
   #                 ranges[component] = self.dataset.attrs[component+'_range'][1] - self.dataset.attrs[component+'_range'][0]
   #             domain = self.dataset.afl.comp.get_range_scaled(ranges=ranges,components=components)
   #         else:
   #             raise ValueError('Domain not recognized!')
   #     else:
   #         domain = self.dataset.afl.comp.get(components=components)
   #     return domain
            
    def transform_domain(self,components=None):
            
        if 'GP_domain_transform' in self.dataset.attrs:
            if self.dataset.attrs['GP_domain_transform']=='ternary':
                if not (len(self.dataset.attrs['components'])==3):   
                    raise ValueError("Ternary domain transform specified but len(components)!=3") 
                domain = self.dataset.afl.comp.ternary_to_xy(components=components)
            elif self.dataset.attrs['GP_domain_transform']=='standard_scaled':
                domain = self.dataset.afl.comp.get_standard_scaled(components=components)
            elif self.dataset.attrs['GP_domain_transform']=='range_scaled':
                components = self.dataset.afl.comp._get_default(components)
                ranges = {}
                for component in components:
                    ranges[component] = {}
                    ranges[component]['min'] = self.dataset.attrs[component+'_range'][0]
                    ranges[component]['max'] = self.dataset.attrs[component+'_range'][1]
                    ranges[component]['range'] = self.dataset.attrs[component+'_range'][1] - self.dataset.attrs[component+'_range'][0]
                domain = self.dataset.afl.comp.get_range_scaled(ranges,components=components)
            else:
                raise ValueError('Domain not recognized!')
        else:
            domain = self.dataset.afl.comp.get(components=components)
        return domain

    def reset_GP(self,dataset,kernel=None,heteroscedastic=False):
    #    """
    #   Constructs the GP model given a likelihood, the input data, a kernel function, and establishes an optimizer
    #   -------------
    #   
    #   """
        self.dataset = dataset
        data = self.construct_data(heteroscedastic)
            
        if kernel is None:
            kernel = gpflow.kernels.Matern32(variance=0.1,lengthscales=0.1) 
        
        if heteroscedastic:
            likelihood = HeteroskedasticGaussian()
            
            #instantiate the model with the data, kernel, and likelihood
            self.model = gpflow.models.VGP(
                data=data, 
                kernel=kernel, 
                likelihood=likelihood, 
                num_latent_gps=1
                )

            #This is from the aforementioned example. 
            #the adam optimizer algorithm is already implemented
            #the natgrad optimizer is necessary for the heteroscedastic regression, gamma is a fixed parameter here. may need to be optimized
            self.natgrad = NaturalGradient(gamma=0.5) 
            self.adam = tf.optimizers.legacy.Adam()
            set_trainable(self.model.q_mu, False)
            set_trainable(self.model.q_sqrt, False)
            
        else:
             self.model = gpflow.models.GPR(
                     data = data,
                     kernel = kernel,
                     )
             self.optimizer = tf.optimizers.legacy.Adam(learning_rate=0.001)
            

    def reset_monitoring(self,log_dir='test/',iter_period=1):
        model_task = ModelToTensorBoard(log_dir, self.model,keywords_to_monitor=['*'])
        lml_task   = ScalarToTensorBoard(log_dir, lambda: self.loss(), "Training Loss")
        
        fast_tasks = MonitorTaskGroup([model_task,lml_task],period=iter_period)
        self.iter_monitor = Monitor(fast_tasks)
        
        image_task = ImageToTensorBoard(
            log_dir, 
            self.plot, 
            "Mean/Variance",
            fig_kw=dict(figsize=(18,6)),
            subplots_kw=dict(nrows=1,ncols=3)
        )
        slow_tasks = MonitorTaskGroup(image_task) 
        self.final_monitor = Monitor(slow_tasks)

    def optimize(self,N=1000, gamma=0.5, final_monitor_step=None,progress_bar=False, tol=1e-4):
        i  = 0
        break_criteria = False
        
        if progress_bar:
            while (i<N) or (break_criteria==True):
                
                pre_step_HPs = np.array([i.numpy() for i in self.model.parameters])
                self._step(i)
                post_step_HPs = np.array([i.numpy() for i in self.model.parameters])
                i+=1
                if all(abs(pre_step_HPs-post_step_HPs) <= tol):
                    break_criteria=True
                    break
            # for i in tqdm.tqdm(tf.range(N),total=N):
            #     self._step(i)
        else:
            while (i<=N) or (break_criteria==True):

                pre_step_HPs = np.array([i.numpy() for i in self.model.parameters])
                self._step(i)
                post_step_HPs = np.array([i.numpy() for i in self.model.parameters])
                i+=1
                if all(abs(pre_step_HPs-post_step_HPs) <= tol):
                    break_criteria=True
                    break
            
        if final_monitor_step is None:
            final_monitor_step = i
        self.final_monitor(final_monitor_step)
           
    def _step(self,i):
        #the optimizers in the heteroscedastic GPFlow example

        if self.heteroscedastic:

            self.natgrad.minimize(self.model.training_loss, [(self.model.q_mu, self.model.q_sqrt)])
            self.adam.minimize(self.model.training_loss, self.model.trainable_variables)
        else:
            self.optimizer.minimize(self.model.training_loss, self.model.trainable_variables)
        self.iter_monitor(i)
    
    def predict(self,components):
        domain = self.transform_domain(components=components)
        self.y = self.model.predict_f(domain) 
        self.y_mean = self.y[0].numpy() 
        self.y_var = self.y[1].numpy() 
        return {'mean':self.y_mean,'var':self.y_var}

    
