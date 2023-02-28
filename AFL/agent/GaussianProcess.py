import numpy as np
import gpflow
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import tqdm
import warnings

from AFL.agent.util import ternary_to_xy

class DummyGP:
    def __init__(self,dataset,kernel=None):
        self.dataset=dataset
        self.kernel=kernel
        
    def predict(self,components):
        domain = self.dataset.afl.comp.get(components)
        self.y_mean = np.zeros_like(domain)
        self.y_var = np.zeros_like(domain)
        return {'mean':self.y_mean,'var':self.y_var}

    
class GP:
    def __init__(self,dataset,kernel=None):
        self.reset_GP(dataset,kernel)
        self.iter_monitor = lambda x: None
        self.final_monitor = lambda x: None
        
    def construct_data(self):
        if 'labels' not in self.dataset:
            raise ValueError('Must have labels variable in Dataset before making GP!')

        if 'labels_ordinal' not in self.dataset:
            self.dataset = self.dataset.afl.labels.make_ordinal()
            
        labels = self.dataset['labels_ordinal'].values
        if len(labels.shape)==1:
            labels = labels[:,np.newaxis]
            
        domain = self.transform_domain()
        
        data = (domain,labels)
        return data
        
    def transform_domain(self,components=None):
            
        if 'GP_domain_transform' in self.dataset.attrs:
            if self.dataset.attrs['GP_domain_transform']=='ternary':
                if not (len(self.dataset.attrs['components']==3)):   
                    raise ValueError("Ternary domain transform specified but len(components)!=3") 
                domain = self.dataset.afl.comp.ternary_to_xy(components=components)
            elif self.dataset.attrs['GP_domain_transform']=='standard_scaled':
                domain = self.dataset.afl.comp.get_standard_scaled(components=components)
            elif self.dataset.attrs['GP_domain_transform']=='range_scaled':
                components = self.dataset.afl.comp._get_default(components)
                ranges = {}
                for component in components:
                    ranges[component] = self.dataset.attrs[component+'_range'][1] - self.dataset.attrs[component+'_range'][0]
                domain = self.dataset.afl.comp.get_range_scaled(ranges=ranges,components=components)
            else:
                raise ValueError('Domain not recognized!')
        else:
            domain = self.dataset.afl.comp.get(components=components)
        return domain
            
    def reset_GP(self,dataset,kernel=None):
        self.dataset = dataset
        self.n_classes = dataset.attrs['n_phases']

        data = self.construct_data()
            
        if kernel is None:
            kernel = gpflow.kernels.Matern32(variance=0.1,lengthscales=0.1) 
            
        invlink = gpflow.likelihoods.RobustMax(self.n_classes)  
        likelihood = gpflow.likelihoods.MultiClass(self.n_classes, invlink=invlink)  
        self.model = gpflow.models.VGP(
            data=data, 
            kernel=kernel, 
            likelihood=likelihood, 
            num_latent_gps=self.n_classes
        ) 
        self.loss = self.model.training_loss_closure(compile=True)
        self.trainable_variables = self.model.trainable_variables
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        
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

    def optimize(self,N,final_monitor_step=None,progress_bar=False):
        if progress_bar:
            for i in tqdm.tqdm(tf.range(N),total=N):
                self._step(i)
        else:
            for i in tf.range(N):
                self._step(i)
            
        if final_monitor_step is None:
            final_monitor_step = i
        self.final_monitor(final_monitor_step)
            
    @tf.function
    def _step(self,i):
        self.optimizer.minimize(self.loss,self.trainable_variables) 
        self.iter_monitor(i)
    
    def predict(self,components):
        domain = self.transform_domain(components=components)
        self.y = self.model.predict_y(domain)
        self.y_mean = self.y[0].numpy() 
        self.y_var = self.y[1].numpy() 
        return {'mean':self.y_mean,'var':self.y_var}

    
